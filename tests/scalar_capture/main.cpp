// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int answer = 42;
  krepe::parallel_for(
      "test_kernel", 1, KOKKOS_LAMBDA(int) {
        Kokkos::printf("Hello from the kernel! The answer is %d\n", answer);
      });
  Kokkos::fence();

  return 0;
}
