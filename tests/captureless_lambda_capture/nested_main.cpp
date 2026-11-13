// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values("values", 1);

  // Inspired by the captureless callable in Ginkgo's assembly example:
  // https://github.com/ginkgo-project/ginkgo/blob/6d4f9df0227999f7feef96a97a166f640d0d21fe/examples/kokkos-assembly/kokkos-assembly.cpp#L209-L217
  const auto f = [] KOKKOS_FUNCTION(int x) { return 6 * x; };
  krepe::parallel_for(
      "test_kernel", 1, KOKKOS_LAMBDA(int i) { values(i) = f(i + 1); });
  Kokkos::fence();

  return 0;
}
