// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;

  Kokkos::parallel_for("test_kernel", Kokkos::RangePolicy<>(0, 1),
                       krepe::replay_functor(KOKKOS_LAMBDA(int) {
                         // Capture the View without dereferencing its zero-byte
                         // allocation.
                         (void)values.data();
                       }));
  Kokkos::fence();

  using memory_space = Kokkos::View<int*>::memory_space;
  int* in_data       = static_cast<int*>(
      krepe::get_allocation<memory_space>("empty_values_without_extent"));
  int* out_data = static_cast<int*>(
      krepe::get_out_allocation<memory_space>("empty_values_without_extent"));

  if (in_data != nullptr || out_data != nullptr) {
    Kokkos::printf(
        "Expected null replay pointers for a view without an explicit "
        "extent\n");
    return 1;
  }

  return 0;
}
