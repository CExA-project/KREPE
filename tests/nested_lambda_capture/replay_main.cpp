// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

#include <stdexcept>
#include <string>
#include <tuple>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  static constexpr int N = 128;
  Kokkos::View<int*> values;
  int multiplier     = 0;
  int shift          = 0;
  int sibling_offset = -3;
  char prefix        = 7;
  double suffix      = -5.0;
  int outer_offset   = -2;
  int view_offset    = -100;

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
  const auto functor = KOKKOS_LAMBDA(int i) { inner(i); };

  krepe::parallel_for("test_kernel", 0, functor);
  Kokkos::fence();

  krepe::compare_views<int*>(
      "values", std::make_tuple(N), [](auto, auto replayed) {
        const auto actual =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), replayed);
        for (int i = 0; i < N; ++i) {
          if (actual(i) != 2 * i + 67) {
            throw std::runtime_error("Nested lambda output differs at " +
                                     std::to_string(i));
          }
        }
      });

  return 0;
}
