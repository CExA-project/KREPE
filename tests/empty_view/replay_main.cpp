// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

#include <cstdlib>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;

  Kokkos::parallel_for("test_kernel", Kokkos::RangePolicy<>(0, 1),
                       krepe::replay_functor(KOKKOS_LAMBDA(int) {
                         if (values.extent(0) != 0) {
                           values(0) = 1;
                         }
                       }));
  Kokkos::fence();

  using memory_space = Kokkos::View<int*>::memory_space;
  int* in_data =
      static_cast<int*>(krepe::get_allocation<memory_space>("empty_values"));
  int* out_data = static_cast<int*>(
      krepe::get_out_allocation<memory_space>("empty_values"));

  if (in_data != nullptr || out_data != nullptr) {
    Kokkos::printf("Expected null replay pointers for an empty view\n");
    return 1;
  }

  krepe::compare_views<int*>(
      "empty_values", std::make_tuple(0),
      [](auto ref_values, auto replay_values) {
        if (ref_values.extent(0) != 0 || replay_values.extent(0) != 0) {
          Kokkos::printf(
              "Expected empty replay views, got extents %zu and %zu\n",
              ref_values.extent(0), replay_values.extent(0));
          std::exit(1);
        }
      });

  return 0;
}
