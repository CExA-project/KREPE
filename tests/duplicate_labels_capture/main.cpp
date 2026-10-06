// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> A("A", N);
  Kokkos::View<int*> B("values", N);
  Kokkos::View<int*> C("values", N);
  Kokkos::parallel_for(
      "init", N, KOKKOS_LAMBDA(int i) {
        A(i) = i;
        B(i) = i % 32;
      });

  krepe::parallel_for(
      "test_kernel", N, KOKKOS_LAMBDA(int i) {
        B(i) *= 3;
        C(i) = A(i) + B(i);
      });
  Kokkos::fence();

  return 0;
}
