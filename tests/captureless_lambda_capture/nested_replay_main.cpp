// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

#include <tuple>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;
  const auto f = [] KOKKOS_FUNCTION(int x) { return 6 * x; };
  krepe::parallel_for(
      "test_kernel", 0, KOKKOS_LAMBDA(int i) { values(i) = f(i + 1); });
  Kokkos::fence();

  krepe::compare_views<int*>(
      "values", std::make_tuple(1), [](auto, auto replayed) {
        const auto h_values =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), replayed);
        if (h_values(0) != 6) {
          Kokkos::abort("Unexpected captureless nested lambda result");
        }
      });

  return 0;
}
