// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values("empty_values", 0);

  Kokkos::parallel_for("test_kernel", Kokkos::RangePolicy<>(0, 1),
                       krepe::replay_functor(KOKKOS_LAMBDA(int) {
                         if (values.extent(0) != 0) {
                           values(0) = 1;
                         }
                       }));
  Kokkos::fence();

  if (values.extent(0) != 0) {
    Kokkos::printf("Expected an empty view, got extent %zu\n",
                   values.extent(0));
    return 1;
  }

  return 0;
}
