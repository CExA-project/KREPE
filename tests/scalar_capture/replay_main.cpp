// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int answer = 0;
  krepe::parallel_for(
      "test_kernel", 0, KOKKOS_LAMBDA(int) {
        Kokkos::printf("Hello from the kernel! The answer is %d\n", answer);
      });
  Kokkos::fence();

  return 0;
}
