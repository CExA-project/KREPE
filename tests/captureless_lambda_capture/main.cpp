// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  krepe::parallel_for("test_kernel", 1, [] KOKKOS_FUNCTION(int i) {
    Kokkos::printf("Captureless lambda: %d\n", i + 1);
  });
  Kokkos::fence();

  return 0;
}
