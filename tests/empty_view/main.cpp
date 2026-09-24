#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> first_values("empty_values", 0);
  Kokkos::View<int*> second_values("empty_values", 0);

  Kokkos::parallel_for("test_kernel", Kokkos::RangePolicy<>(0, 1),
                       krepe::replay_functor(KOKKOS_LAMBDA(int) {
                         if (first_values.extent(0) != 0) {
                           first_values(0) = 1;
                         }
                         if (second_values.extent(0) != 0) {
                           second_values(0) = 2;
                         }
                       }));
  Kokkos::fence();

  if (first_values.extent(0) != 0 || second_values.extent(0) != 0) {
    Kokkos::printf("Expected empty views, got extents %zu and %zu\n",
                   first_values.extent(0), second_values.extent(0));
    return 1;
  }

  return 0;
}
