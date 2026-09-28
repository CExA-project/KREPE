// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  auto policy = Kokkos::TeamPolicy(4, Kokkos::AUTO);
  krepe::parallel_for(
      "test_kernel", policy, KOKKOS_LAMBDA(decltype(policy)::member_type team) {
        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(team, team.league_rank() * 256,
                                    (team.league_rank() + 1) * 256),
            [&](int i) { values(i) *= team.team_size() + team.league_size(); });
      });
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(5) = %d\n", h_values(5));

  return 0;
}
