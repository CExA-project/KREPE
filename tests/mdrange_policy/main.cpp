// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int M = 32;
  const int N = 64;
  const int K = 16;
  Kokkos::View<int***> values("values", M, N, K);
  Kokkos::parallel_for(
      "init", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {M, N, K}),
      KOKKOS_LAMBDA(int i, int j, int k) { values(i, j, k) = i * j * k; });

  krepe::parallel_for(
      "test_kernel",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {M, N, K}),
      KOKKOS_LAMBDA(int i, int j, int k) { values(i, j, k) *= 2; });
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(1, 2, 3) = %d\n", h_values(1, 2, 3));

  return 0;
}
