// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int M = 32;
  const int N = 32;
  Kokkos::View<int**> values("values", M, N);
  Kokkos::parallel_for(
      "init", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}),
      KOKKOS_LAMBDA(int i, int j) { values(i, j) = i * j; });

  using exec_space = Kokkos::DefaultExecutionSpace;
  krepe::parallel_for(
      "test_kernel",
      Kokkos::MDRangePolicy<
          Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Right>,
          exec_space>({0, 0}, {M, N}, {4, 4}),
      KOKKOS_LAMBDA(int i, int j) {
        values(i, j) *= 2;
#if defined(KOKKOS_ENABLE_SERIAL)
        if constexpr (std::is_same_v<exec_space, Kokkos::Serial>) {
          Kokkos::printf("%d %d\n", i, j);
        }
#endif
      });
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(1, 2) = %d\n", h_values(1, 2));

  return 0;
}
