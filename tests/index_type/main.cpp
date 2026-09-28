// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  krepe::parallel_for(
      "test_kernel",
      Kokkos::RangePolicy<Kokkos::IndexType<std::uint64_t>>(0, N),
  // nvcc doesn't support generic host device lambdas
#if !defined(KOKKOS_COMPILER_NVCC)
      KOKKOS_LAMBDA(std::integral auto i) {
        if (!std::is_same_v<decltype(i), std::uint64_t>) {
          Kokkos::abort("Failed");
        }
#else
      KOKKOS_LAMBDA(std::uint64_t i) {
#endif
        values(i) *= 2;
      });
  Kokkos::fence();

  auto h_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  Kokkos::printf("values(5) = %d\n", h_values(5));

  return 0;
}
