// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  krepe::parallel_for(
      "test_kernel", 0,
      KOKKOS_LAMBDA(int) { Kokkos::printf("Hello from the kernel!\n"); });
  Kokkos::fence();

  return 0;
}
