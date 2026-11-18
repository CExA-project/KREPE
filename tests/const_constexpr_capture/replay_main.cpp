// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  volatile int input      = 0;
  const int runtime_value = input;
  const int multiplier    = 6;
  constexpr int offset    = 7;

  const auto functor = KOKKOS_LAMBDA(int i) {
    const int result = runtime_value * multiplier + offset + i;
    Kokkos::printf("const/constexpr result: %d\n", result);
    return result;
  };
  krepe::parallel_for("test_kernel", 0, functor);
  Kokkos::fence();

  const auto replayed = krepe::replay_functor(functor);
  if (replayed(1) != 26) {
    Kokkos::abort("The runtime const value was not restored");
  }

  return 0;
}
