#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  bool rejected = false;
  try {
    replay_scope.reset_inputs();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("Reset without saved inputs was accepted");
  }

  const int N = 1024;
  Kokkos::View<int*> values;
  // Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  krepe::parallel_for(
      "test_kernel", 0, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  // auto h_values =
  //     Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  // Kokkos::printf("values(5) = %d\n", h_values(5));

  krepe::compare_views<int*>(
      "values", std::make_tuple(1024), [](auto ref_values, auto replay_values) {
        auto h_replay_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), replay_values);

        auto h_ref_values = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), ref_values);

        for (int i = 0; i < N; i++) {
          if (h_replay_values(i) != h_ref_values(i)) {
            Kokkos::printf("At index %d, expected %d but got %d\n", i,
                           h_ref_values(i), h_replay_values(i));
            std::exit(1);
          }
        }
      });

  return 0;
}
