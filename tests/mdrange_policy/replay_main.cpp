// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int M = 32;
  const int N = 64;
  const int K = 16;
  Kokkos::View<int***> values;
  // Kokkos::parallel_for(
  //     "init", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {M, N, K}),
  //     KOKKOS_LAMBDA(int i, int j, int k) { values(i, j, k) = i * j * k; });

  krepe::parallel_for(
      "test_kernel",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {1, 1, 1}),
      KOKKOS_LAMBDA(int i, int j, int k) { values(i, j, k) *= 2; });
  Kokkos::fence();

  // auto h_values =
  //     Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  // Kokkos::printf("values(1, 2, 3) = %d\n", h_values(1, 2, 3));

  krepe::compare_views<int***>(
      "values", std::make_tuple(M, N, K),
      [](auto ref_values, auto replay_values) {
        auto h_replay_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), replay_values);

        auto h_ref_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), ref_values);

        for (int i = 0; i < M; i++) {
          for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
              if (h_replay_values(i, j, k) != h_ref_values(i, j, k)) {
                Kokkos::printf(
                    "At index (%d, %d, %d), expected %d but got %d\n", i, j, k,
                    h_ref_values(i, j, k), h_replay_values(i, j, k));
                std::exit(1);
              }
            }
          }
        }
      });

  return 0;
}
