// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int M = 32;
  const int N = 64;
  const int K = 16;
  Kokkos::View<int***> values("values", M, N, K);
  Kokkos::parallel_for(
      "init", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {M, N, K}),
      KOKKOS_LAMBDA(int i, int j, int k) { values(i, j, k) = i * j * k; });

  krepe::parallel_for(
      "test_kernel",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<std::uint64_t>>(
          {0, 0, 0}, {M, N, K}),
#if !defined(KOKKOS_COMPILER_NVCC)
      KOKKOS_LAMBDA(std::integral auto i, std::integral auto j,
                    std::integral auto k) {
        if constexpr (!std::is_same_v<decltype(i), std::uint64_t> ||
                      !std::is_same_v<decltype(j), std::uint64_t> ||
                      !std::is_same_v<decltype(k), std::uint64_t>) {
          Kokkos::abort("Fail");
        }
        values(i, j, k) *= 2;
      });
#else
      KOKKOS_LAMBDA(std::uint64_t i, std::uint64_t j, std::uint64_t k) {
        values(i, j, k) *= 2;
      });
#endif
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(1, 2, 3) = %d\n", h_values(1, 2, 3));

  return 0;
}
