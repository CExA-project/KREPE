// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

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
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  int x = 42;
  int y = -67;
  krepe::parallel_for("test_kernel", 1, Functor(x, y));
  Kokkos::fence();

  return 0;
}
