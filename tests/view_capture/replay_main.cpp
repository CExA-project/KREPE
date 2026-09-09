#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>
#include <stdexcept>
#include <tuple>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N   = 1024;
  using Triplet = Kokkos::Array<float, 3>;
  using Space   = Kokkos::DefaultExecutionSpace::memory_space;
  Kokkos::View<int*> values;
  Kokkos::View<Triplet*> triplets;
  // Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  krepe::parallel_for(
      "test_kernel", 0, KOKKOS_LAMBDA(int i) {
        values(i) *= 2;
        if (i == 0) {
          for (int j = 0; j < 3; ++j) triplets(0)[j] += 1.0f;
        }
      });
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

  const auto triplet_allocations = krepe::get_allocations<Space>("triplets");
  if (triplet_allocations.size() != 1) {
    Kokkos::printf("Expected one allocation named triplets\n");
    return 1;
  }

  const auto& allocation = triplet_allocations.front();
  // Kokkos may round this 12-byte value up to a 16-byte allocation. Explicit
  // dimensions must compare the logical value without consuming the padding.
  const bool matches = krepe::compare_views<Triplet*, Space>(
      allocation, std::make_tuple(1), [](auto reference, auto actual) {
        auto h_reference =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reference);
        auto h_actual =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
        for (int j = 0; j < 3; ++j) {
          if (h_reference(0)[j] != float(j + 2) ||
              h_actual(0)[j] != h_reference(0)[j]) {
            return false;
          }
        }
        return true;
      });
  if (!matches) {
    Kokkos::printf("Explicitly shaped triplet comparison failed\n");
    return 1;
  }

  if (allocation.size_bytes % sizeof(Triplet) != 0) {
    bool callback_called = false;
    bool rejected        = false;
    try {
      krepe::compare_views<Triplet*, Space>(
          allocation, [&](auto, auto) { callback_called = true; });
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    if (!rejected || callback_called) {
      Kokkos::printf("Inferred triplet length must reject partial elements\n");
      return 1;
    }
  }

  bool callback_called = false;
  bool rejected        = false;
  try {
    const auto too_many = allocation.size_bytes / sizeof(Triplet) + 1;
    krepe::compare_views<Triplet*, Space>(
        allocation, std::make_tuple(too_many),
        [&](auto, auto) { callback_called = true; });
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected || callback_called) {
    Kokkos::printf("Explicit triplet dimensions must fit the allocation\n");
    return 1;
  }

  return 0;
}
