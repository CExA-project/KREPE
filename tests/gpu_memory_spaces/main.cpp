// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

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

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  // Kokkos currently uses cudaMallocAsync for CudaSpace allocations of at
  // least 40 kB when its asynchronous allocator is enabled. Keep this view
  // larger than that threshold so the same test covers that configuration.
  constexpr int N = 16 * 1024;

  Kokkos::View<int*, DeviceSpace> device_values("device_values", N);
  Kokkos::View<int*, ManagedSpace> managed_values("managed_values", N);
  Kokkos::View<int*, HostPinnedSpace> host_pinned_values("host_pinned_values",
                                                         N);

  Kokkos::parallel_for(
      "init", N, KOKKOS_LAMBDA(const int i) {
        device_values(i)      = i + 1;
        managed_values(i)     = i + 2;
        host_pinned_values(i) = i + 3;
      });

  krepe::parallel_for(
      "test_kernel", N, KOKKOS_LAMBDA(const int i) {
        device_values(i) *= 2;
        managed_values(i) *= 3;
        host_pinned_values(i) *= 4;
      });
  Kokkos::fence();

  return 0;
}
