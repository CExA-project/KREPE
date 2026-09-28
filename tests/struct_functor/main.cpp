// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

struct Functor {
 private:
  using view_t = Kokkos::View<int*>;

  int factor;
  view_t values;

 public:
  Functor(int factor, view_t values) : factor(factor), values(values) {}

  KOKKOS_FUNCTION void operator()(int i) const { values(i) *= factor; }
};

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  int factor = 3;
  krepe::parallel_for("test_kernel", N, Functor(factor, values));
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(5) = %d\n", h_values(5));

  return 0;
}
