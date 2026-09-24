#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

#include <cstdlib>
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

void compare_values(const char* label) {
  // The replayer reserves all non-host allocations in device virtual memory.
  // Use DeviceSpace here even for the managed and host-pinned source views so
  // validation accesses the replay allocation through the correct path.
  krepe::compare_views<int*, DeviceSpace>(
      label, std::make_tuple(16 * 1024),
      [label](const auto expected, const auto actual) {
        auto host_expected =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expected);
        auto host_actual =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
        for (std::size_t i = 0; i < expected.extent(0); ++i) {
          if (host_actual(i) != host_expected(i)) {
            Kokkos::printf("%s: at index %zu, expected %d but got %d\n", label,
                           i, host_expected(i), host_actual(i));
            std::exit(1);
          }
        }
      });
}

}  // namespace

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv, true);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*, DeviceSpace> device_values;
  Kokkos::View<int*, ManagedSpace> managed_values;
  Kokkos::View<int*, HostPinnedSpace> host_pinned_values;

  for (int repetition = 0; repetition < 3; ++repetition) {
    replay_scope.reset_inputs();
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

    compare_values("device_values");
    compare_values("managed_values");
    compare_values("host_pinned_values");
  }

  return 0;
}
