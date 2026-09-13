#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> first_values;
  Kokkos::View<int*> second_values;

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

  using memory_space     = Kokkos::View<int*>::memory_space;
  const auto allocations = krepe::get_allocations<memory_space>("empty_values");
  if (allocations.size() != 2) {
    Kokkos::printf("Expected two empty allocations sharing one label\n");
    return 1;
  }

  for (const auto& allocation : allocations) {
    if (allocation.data != nullptr || allocation.reference_data != nullptr ||
        allocation.size_bytes != 0 || allocation.reference_size_bytes != 0 ||
        !allocation.has_input || !allocation.has_reference) {
      Kokkos::printf("Expected null replay pointers for an empty view\n");
      return 1;
    }

    int callback_count = 0;
    const bool matches = krepe::compare_views<int*>(
        allocation, std::make_tuple(0),
        [&](auto ref_values, auto replay_values) {
          ++callback_count;
          return ref_values.extent(0) == 0 && replay_values.extent(0) == 0;
        });
    if (!matches || callback_count != 1) {
      Kokkos::printf("Expected one comparison of empty views per allocation\n");
      return 1;
    }
  }

  return 0;
}
