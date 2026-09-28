// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  using exec_space   = Kokkos::DefaultHostExecutionSpace;
  using memory_space = typename exec_space::memory_space;
  const int N        = 1024;
  Kokkos::View<int*, memory_space> values("values", N);
  Kokkos::parallel_for(
      "init", Kokkos::RangePolicy<exec_space>(0, N),
      KOKKOS_LAMBDA(int i) { values(i) = i; });

  krepe::parallel_for(
      "test_kernel", Kokkos::RangePolicy<exec_space>(0, N),
      KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  Kokkos::printf("values(5) = %d\n", values(5));

  return 0;
}
