#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>

#if defined(KOKKOS_ENABLE_CUDA)
using DeviceSpace     = Kokkos::CudaSpace;
using ManagedSpace    = Kokkos::CudaUVMSpace;
using HostPinnedSpace = Kokkos::CudaHostPinnedSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using DeviceSpace     = Kokkos::HIPSpace;
using ManagedSpace    = Kokkos::HIPManagedSpace;
using HostPinnedSpace = Kokkos::HIPHostPinnedSpace;
#else
#error "gpu_memory_spaces requires either the CUDA or HIP backend"
#endif

namespace {

template <class MemorySpace>
void require_dumped_allocation(const char* label) {
  if (krepe::get_allocation<MemorySpace>(label) == nullptr ||
      krepe::get_out_allocation<MemorySpace>(label) == nullptr) {
    Kokkos::printf("Expected captured bytes for allocation \"%s\"\n", label);
    std::exit(1);
  }
}

template <class MemorySpace>
krepe::ReplayAllocation require_descriptor(const char* label) {
  const auto allocations = krepe::get_allocations<MemorySpace>(label);
  if (allocations.size() != 1 ||
      allocations.front().memory_space != MemorySpace::name()) {
    Kokkos::printf("Expected one %s allocation named %s\n", MemorySpace::name(),
                   label);
    std::exit(1);
  }
  for (const char* other :
       {"device_values", "managed_values", "host_pinned_values"}) {
    if (std::string(label) != other &&
        !krepe::get_allocations<MemorySpace>(other).empty()) {
      Kokkos::printf("%s must not match captured space %s\n", other,
                     MemorySpace::name());
      std::exit(1);
    }
  }
  std::size_t matching_labels = 0;
  for (const auto& allocation : krepe::get_allocations<MemorySpace>()) {
    if (allocation.memory_space != MemorySpace::name()) {
      Kokkos::printf("Enumeration returned an incompatible captured space\n");
      std::exit(1);
    }
    if (allocation.label == label) ++matching_labels;
  }
  if (matching_labels != 1) std::exit(1);
  return allocations.front();
}

template <class MemorySpace>
void require_comparison_rejected(const krepe::ReplayAllocation& allocation) {
  bool callback_called = false;
  bool rejected        = false;
  try {
    krepe::compare_views<int*, MemorySpace>(
        allocation, [&](auto, auto) { callback_called = true; });
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected || callback_called) {
    Kokkos::printf("Replay allocation %s must not be accessed as %s\n",
                   allocation.label.c_str(), MemorySpace::name());
    std::exit(1);
  }
}

void compare_values(const krepe::ReplayAllocation& allocation) {
  const char* label = allocation.label.c_str();
  // The replayer reserves all non-host allocations in device virtual memory.
  // Use DeviceSpace here even for the managed and host-pinned source views so
  // validation accesses the replay allocation through the correct path.
  const auto compare = [label](const auto expected, const auto actual) {
    auto host_expected =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected);
    auto host_actual =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
    for (std::size_t i = 0; i < expected.extent(0); ++i) {
      if (host_actual(i) != host_expected(i)) {
        Kokkos::printf("%s: at index %zu, expected %d but got %d\n", label, i,
                       host_expected(i), host_actual(i));
        std::exit(1);
      }
    }
  };
  krepe::compare_views<int*, DeviceSpace>(label, std::make_tuple(16 * 1024),
                                          compare);
  krepe::compare_views<int*, DeviceSpace>(allocation, compare);
}

}  // namespace

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*, DeviceSpace> device_values;
  Kokkos::View<int*, ManagedSpace> managed_values;
  Kokkos::View<int*, HostPinnedSpace> host_pinned_values;

  krepe::parallel_for(
      "test_kernel", 0, KOKKOS_LAMBDA(const int i) {
        device_values(i) *= 2;
        managed_values(i) *= 3;
        host_pinned_values(i) *= 4;
      });
  Kokkos::fence();

  require_dumped_allocation<DeviceSpace>("device_values");
  require_dumped_allocation<ManagedSpace>("managed_values");
  require_dumped_allocation<HostPinnedSpace>("host_pinned_values");

  const auto device  = require_descriptor<DeviceSpace>("device_values");
  const auto managed = require_descriptor<ManagedSpace>("managed_values");
  const auto pinned = require_descriptor<HostPinnedSpace>("host_pinned_values");
  for (const auto& allocation : {device, managed, pinned}) {
    require_comparison_rejected<ManagedSpace>(allocation);
    require_comparison_rejected<HostPinnedSpace>(allocation);
    require_comparison_rejected<Kokkos::HostSpace>(allocation);
    compare_values(allocation);
  }

  return 0;
}
