// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

struct Functor {
 private:
  int x;
  int y;

 public:
  Functor(int x, int y) : x(x), y(y) {}

  KOKKOS_FUNCTION void operator()(int) const {
    Kokkos::printf("x is %d, y is %d\n", x, y);
  }
};

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int x = 0;
  int y = 0;
  krepe::parallel_for("test_kernel", 1, Functor(x, y));
  Kokkos::fence();

  return 0;
}
