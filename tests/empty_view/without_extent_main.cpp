// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values("empty_values_without_extent");

  Kokkos::parallel_for("test_kernel", Kokkos::RangePolicy<>(0, 1),
                       krepe::replay_functor(KOKKOS_LAMBDA(int) {
                         // Capture the View without dereferencing its zero-byte
                         // allocation.
                         (void)values.data();
                       }));
  Kokkos::fence();

  return 0;
}
