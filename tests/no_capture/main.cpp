// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  krepe::parallel_for(
      "test_kernel", 1,
      KOKKOS_LAMBDA(int) { Kokkos::printf("Hello from the kernel!\n"); });
  Kokkos::fence();

  return 0;
}
