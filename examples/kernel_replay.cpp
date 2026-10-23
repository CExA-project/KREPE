// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <krepe/replayer.hpp>

#include <iostream>
#include <tuple>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;

  // Replay with the captured inputs and policy (the 0 argument is ignored).
  krepe::parallel_for(
      "scale", 0, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  // Compare the replayed values with the captured output.
  krepe::compare_views<int*>(
      "values", std::make_tuple(1024), [](auto expected, auto actual) {
        if (!Kokkos::Experimental::equal(Kokkos::DefaultExecutionSpace(),
                                         expected, actual)) {
          Kokkos::abort("Replay verification failed");
        }
      });
  std::cout << "Replay verified: 1024 values match.\n";

  return 0;
}
