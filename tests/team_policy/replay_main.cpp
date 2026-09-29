// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values;
  // Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  auto policy = Kokkos::TeamPolicy(1, 1);
  krepe::parallel_for(
      "test_kernel", policy, KOKKOS_LAMBDA(decltype(policy)::member_type team) {
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, team.league_rank() * 256,
                                    (team.league_rank() + 1) * 256),
            [&](int i) { values(i) *= team.team_size() + team.league_size(); });
      });
  Kokkos::fence();

  // auto h_values =
  //     Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  // Kokkos::printf("values(5) = %d\n", h_values(5));

  krepe::compare_views<int*>(
      "values", std::make_tuple(1024), [](auto ref_values, auto replay_values) {
        auto h_replay_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), replay_values);

        auto h_ref_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), ref_values);

        for (int i = 0; i < N; i++) {
          if (h_replay_values(i) != h_ref_values(i)) {
            Kokkos::printf("At index %d, expected %d but got %d\n", i,
                           h_ref_values(i), h_replay_values(i));
            std::exit(1);
          }
        }
      });

  return 0;
}
