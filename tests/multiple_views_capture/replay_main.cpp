#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> A;
  Kokkos::View<int*> B;
  Kokkos::View<int*> C;
  // Kokkos::parallel_for(
  //     "init", N, KOKKOS_LAMBDA(int i) {
  //       A(i) = i;
  //       B(i) = i % 32;
  // });

  krepe::parallel_for(
      "test_kernel", N, KOKKOS_LAMBDA(int i) { C(i) = A(i) + B(i); });
  Kokkos::fence();

  // auto h_C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
  // Kokkos::printf("C(5) = %d\n", h_C(5));

  krepe::compare_views<int*>(
      "C", std::make_tuple(1024), [](auto ref_C, auto replay_C) {
        auto h_replay_C =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), replay_C);
        auto h_ref_C =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_C);

        for (int i = 0; i < N; i++) {
          if (h_replay_C(i) != h_ref_C(i)) {
            Kokkos::printf("At index %d, expected %d but got %d\n", i,
                           h_ref_C(i), h_replay_C(i));
            std::exit(1);
          }
        }
      });

  return 0;
}
