// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>
#include <string>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  krepe::add_metadata("foo", "bar");
  krepe::add_metadata("with null bytes", std::string("hello\0world", 12));
  int n_iter = 10;
  krepe::add_metadata("n_iter", std::to_string(n_iter));

  int sum = 0;
  Kokkos::parallel_reduce(
      "test_kernel", n_iter,
      krepe::replay_functor(KOKKOS_LAMBDA(int, int& sum) { sum++; }), sum);

  Kokkos::printf("Result is %d\n", sum);

  return 0;
}
