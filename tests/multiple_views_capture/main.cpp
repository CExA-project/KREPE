#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> A("A", N);
  Kokkos::View<int*> B("B", N);
  Kokkos::View<int*> C("C", N);
  Kokkos::parallel_for(
      "init", N, KOKKOS_LAMBDA(int i) {
        A(i) = i;
        B(i) = i % 32;
      });

  krepe::parallel_for(
      "test_kernel", N, KOKKOS_LAMBDA(int i) { C(i) = A(i) + B(i); });
  Kokkos::fence();

  auto h_C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
  Kokkos::printf("C(5) = %d\n", h_C(5));

  return 0;
}
