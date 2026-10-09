// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

#include <stdexcept>

#include "aligned_capture.hpp"

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int multiplier     = 3;
  int shift          = 7;
  int sibling_offset = 11;
  char prefix        = 2;
  double suffix      = 13.0;
  int outer_offset   = 5;
  AlignedCapture aligned{19};

  const auto leaf    = KOKKOS_LAMBDA(int i) { return multiplier * i + shift; };
  const auto sibling = KOKKOS_LAMBDA(int i) { return sibling_offset - i; };
  const auto empty   = KOKKOS_LAMBDA(int i) { return i + 1; };
  const auto middle  = KOKKOS_LAMBDA(int i) {
    int before = prefix;
    int result = leaf(i) + sibling(i) + empty(i);
    return before + result + static_cast<int>(suffix) + aligned.value;
  };
  auto functor = KOKKOS_LAMBDA(int i) {
    int before = outer_offset;
    return before + middle(i) + sibling(i);
  };
  krepe::parallel_for(
      "test_kernel",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, 1), functor);
  Kokkos::fence();

  if (functor(17) != 103) {
    throw std::runtime_error("Unexpected reference nested lambda result");
  }

  return 0;
}
