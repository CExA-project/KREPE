// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <iostream>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  constexpr int number_of_values = 1024;
  Kokkos::View<int*> values("view", number_of_values);

  Kokkos::parallel_for(
      "fill_values", Kokkos::RangePolicy<>(0, number_of_values),
      KOKKOS_LAMBDA(const int i) { values(i) = i; });

  int factor = 2;

  int sum = 0;
  Kokkos::parallel_reduce(
      "sum_values", Kokkos::RangePolicy<>(0, number_of_values),
      krepe::replay_functor(KOKKOS_LAMBDA(const int i, int& partial_sum) {
        partial_sum += values(i) * factor;
      }),
      sum);

  constexpr int expected_sum =
      2 * number_of_values * (number_of_values - 1) / 2;
  if (sum != expected_sum) {
    std::cerr << "Unexpected Kokkos reduction result: got " << sum
              << ", expected " << expected_sum << '\n';
    return 1;
  }

  std::cout << "Kokkos kernel reduction succeeded: " << sum << '\n';
  return 0;
}
