// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>
#include <stdexcept>
#include <tuple>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N   = 1024;
  using Triplet = Kokkos::Array<int, 3>;
  using Space   = Kokkos::DefaultExecutionSpace::memory_space;
  Kokkos::View<int*> values;
  Kokkos::View<Triplet*> triplets;

  krepe::parallel_for(
      "test_kernel", 0, KOKKOS_LAMBDA(int i) {
        values(i) *= 2;
        if (i == 0) {
          for (int j = 0; j < 3; ++j) triplets(0)[j] += 1;
        }
      });
  Kokkos::fence();

  const auto compare = [](auto ref_values, auto replay_values) {
    auto h_replay_values =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), replay_values);

    auto h_ref_values =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ref_values);

    for (int i = 0; i < N; i++) {
      if (h_replay_values(i) != h_ref_values(i)) {
        Kokkos::printf("At index %d, expected %d but got %d\n", i,
                       h_ref_values(i), h_replay_values(i));
        std::exit(1);
      }
    }
  };
  krepe::compare_views<int*>("values", std::make_tuple(N), compare);
  Kokkos::View<int*> named_values("values", 0);
  krepe::compare_views(named_values, std::make_tuple(N), compare);

  const auto triplet_allocations = krepe::get_allocations<Space>("triplets");
  if (triplet_allocations.size() != 1) {
    Kokkos::printf("Expected one allocation named triplets\n");
    return 1;
  }

  const auto& allocation      = triplet_allocations.front();
  bool reference_unchanged    = true;
  const auto compare_triplets = [&](auto reference, auto actual) {
    auto h_reference =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reference);
    auto h_actual =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
    bool matches = true;
    for (int j = 0; j < 3; ++j) {
      if (h_reference(0)[j] != j + 2) {
        reference_unchanged = false;
      }
      if (h_actual(0)[j] != h_reference(0)[j]) {
        matches = false;
      }
    }
    return matches;
  };
  const bool matches = krepe::compare_views<Triplet*, Space>(
      allocation, std::make_tuple(1), compare_triplets);
  if (!matches || !reference_unchanged) {
    Kokkos::printf("Explicitly shaped triplet comparison failed\n");
    return 1;
  }

  // Changing the replayed data must leave the reference intact and compare
  // false.
  using UnmanagedTripletView =
      Kokkos::View<Triplet*, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  UnmanagedTripletView replayed_triplet(static_cast<Triplet*>(allocation.data),
                                        1);
  Kokkos::parallel_for(
      "modify_replayed_triplet", 1,
      KOKKOS_LAMBDA(int) { replayed_triplet(0)[0] += 1; });
  Kokkos::fence();
  const bool still_matches = krepe::compare_views<Triplet*, Space>(
      allocation, std::make_tuple(1), compare_triplets);
  if (still_matches || !reference_unchanged) {
    Kokkos::printf(
        "Changed replay data must compare false with an unchanged reference\n");
    return 1;
  }

  // Force a partial element regardless of allocation padding.
  auto invalid_size                 = allocation;
  invalid_size.size_bytes           = sizeof(Triplet) - 1;
  invalid_size.reference_size_bytes = invalid_size.size_bytes;

  bool callback_called = false;
  bool rejected        = false;
  try {
    krepe::compare_views<Triplet*, Space>(
        invalid_size, [&](auto, auto) { callback_called = true; });
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  if (!rejected || callback_called) {
    Kokkos::printf("Inferred triplet length must reject partial elements\n");
    return 1;
  }

  callback_called = false;
  rejected        = false;
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
