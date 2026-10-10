// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  constexpr int N = 128;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", N, KOKKOS_LAMBDA(int i) { values(i) = -1; });

  int multiplier     = 3;
  int shift          = 7;
  int sibling_offset = 11;
  char prefix        = 2;
  double suffix      = 13.0;
  int outer_offset   = 5;
  int view_offset    = 17;

  const auto leaf    = KOKKOS_LAMBDA(int i) { return multiplier * i + shift; };
  const auto sibling = KOKKOS_LAMBDA(int i) { return sibling_offset - i; };
  const auto empty   = KOKKOS_LAMBDA(int i) { return i + 1; };
  const auto middle  = KOKKOS_LAMBDA(int i) {
    int before = prefix;
    int result = leaf(i) + sibling(i) + empty(i);
    return before + result + static_cast<int>(suffix);
  };
  const auto scalar = KOKKOS_LAMBDA(int i) {
    int before = outer_offset;
    return before + middle(i) + sibling(i);
  };
  const auto inner = KOKKOS_LAMBDA(int i) {
    values(i) = scalar(i) + view_offset;
  };
  auto functor = KOKKOS_LAMBDA(int i) { inner(i); };
  krepe::parallel_for("test_kernel", N, functor);
  Kokkos::fence();

  return 0;
}
