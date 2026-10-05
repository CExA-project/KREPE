// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N   = 1024;
  using Triplet = Kokkos::Array<int, 3>;
  Kokkos::View<int*> values("values", N);
  Kokkos::View<Triplet*> triplets("triplets", 1);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) {
        values(i) = i;
        if (i == 0) {
          for (int j = 0; j < 3; ++j) triplets(0)[j] = j + 1;
        }
      });

  krepe::parallel_for(
      "test_kernel", N, KOKKOS_LAMBDA(int i) {
        values(i) *= 2;
        if (i == 0) {
          for (int j = 0; j < 3; ++j) triplets(0)[j] += 1;
        }
      });
  Kokkos::fence();

  return 0;
}
