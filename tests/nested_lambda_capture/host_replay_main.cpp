// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

#include <stdexcept>

#include "aligned_capture.hpp"

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int multiplier     = 0;
  int shift          = 0;
  int sibling_offset = -3;
  char prefix        = 7;
  double suffix      = -5.0;
  int outer_offset   = -2;
  AlignedCapture aligned{-9};

  const auto leaf    = KOKKOS_LAMBDA(int i) { return multiplier * i + shift; };
  const auto sibling = KOKKOS_LAMBDA(int i) { return sibling_offset - i; };
  const auto empty   = KOKKOS_LAMBDA(int i) { return i + 1; };
  const auto middle  = KOKKOS_LAMBDA(int i) {
    int before = prefix;
    int result = leaf(i) + sibling(i) + empty(i);
    return before + result + static_cast<int>(suffix) + aligned.value;
  };
  const auto functor = KOKKOS_LAMBDA(int i) {
    int before = outer_offset;
    return before + middle(i) + sibling(i);
  };

  constexpr int placeholder_result = -31;

  const auto first = krepe::replay_functor(functor);
  if (first(17) != 103) {
    throw std::runtime_error("Nested host closures were not restored");
  }

  // Reject an incompatible functor after collecting its nested host closures.
  // Unwinding must leave the source closures and View tracking intact.
  Kokkos::Array<int, sizeof(functor) + 1> padding{};
  padding[0]           = 1;
  const auto oversized = KOKKOS_LAMBDA(int i) {
    return functor(i) + padding[0];
  };
  bool rejected = false;
  try {
    (void)krepe::replay_functor(oversized);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected || oversized(17) != placeholder_result + 1 ||
      functor(17) != placeholder_result ||
      !Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_enabled()) {
    throw std::runtime_error("Failed replay did not restore temporary state");
  }

  return 0;
}
