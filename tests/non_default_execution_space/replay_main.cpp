// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  using exec_space   = Kokkos::DefaultHostExecutionSpace;
  using memory_space = typename exec_space::memory_space;
  const int N        = 1024;
  Kokkos::View<int*, memory_space> values;
  // Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  krepe::parallel_for(
      "test_kernel", Kokkos::RangePolicy<exec_space>(0, 1),
      KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  // Kokkos::printf("values(5) = %d\n", values(5));

  krepe::compare_views<int*, memory_space>(
      "values", std::make_tuple(1024), [](auto ref_values, auto replay_values) {
        for (int i = 0; i < N; i++) {
          if (replay_values(i) != ref_values(i)) {
            Kokkos::printf("At index %d, expected %d but got %d\n", i,
                           ref_values(i), replay_values(i));
            std::exit(1);
          }
        }
      });

  return 0;
}
