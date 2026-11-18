// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  // Volatile read prevents this const from being a constant expression.
  volatile int input      = 3;
  const int runtime_value = input;
  const int multiplier    = 6;
  constexpr int offset    = 7;

  const auto functor = KOKKOS_LAMBDA(int i) {
    const int result = runtime_value * multiplier + offset + i;
    Kokkos::printf("const/constexpr result: %d\n", result);
    return result;
  };
  krepe::parallel_for("test_kernel", 1, functor);
  Kokkos::fence();

  if (functor(1) != 26) {
    Kokkos::abort("Unexpected const/constexpr reference result");
  }

  return 0;
}
