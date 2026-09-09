#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

#include <stdexcept>
#include <type_traits>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> A;
  Kokkos::View<int*> B;
  Kokkos::View<int*> C;
  krepe::parallel_for(
      "test_kernel", N, KOKKOS_LAMBDA(int i) {
        B(i) *= 3;
        C(i) = A(i) + B(i);
      });
  Kokkos::fence();

  using memory_space     = Kokkos::DefaultExecutionSpace::memory_space;
  const auto allocations = krepe::get_allocations<memory_space>("values");
  if (allocations.size() != 2 || allocations[0].data == allocations[1].data ||
      allocations[0].reference_data == allocations[1].reference_data ||
      !krepe::get_allocations<memory_space>("missing").empty()) {
    Kokkos::printf("Expected two distinct allocations with the same label\n");
    return 1;
  }
  // Label-only lookup must reject ambiguous allocations.
  try {
    krepe::get_allocation<memory_space>("values");
    return 1;
  } catch (const std::runtime_error&) {
  }
  try {
    krepe::get_out_allocation<memory_space>("values");
    return 1;
  } catch (const std::runtime_error&) {
  }
  std::size_t same_label_count = 0;
  for (const auto& allocation : krepe::get_allocations<memory_space>()) {
    same_label_count += allocation.label == "values";
  }
  if (same_label_count != 2) {
    return 1;
  }

  for (const auto& allocation : allocations) {
    if (!allocation.has_input || !allocation.has_reference ||
        allocation.size_bytes != N * sizeof(int) ||
        allocation.reference_size_bytes != allocation.size_bytes) {
      return 1;
    }
    const bool matches =
        krepe::compare_views<int*>(allocation, [](auto expected, auto actual) {
          static_assert(
              std::is_const_v<typename decltype(expected)::value_type>);
          auto host_actual =
              Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actual);
          auto host_expected = Kokkos::create_mirror_view_and_copy(
              Kokkos::HostSpace(), expected);
          if (actual.extent(0) != N || expected.extent(0) != N) {
            return false;
          }
          for (int i = 0; i < N; i++) {
            if (host_actual(i) != host_expected(i)) {
              Kokkos::printf("At index %d, expected %d but got %d\n", i,
                             host_expected(i), host_actual(i));
              return false;
            }
          }
          return true;
        });
    if (!matches) {
      return 1;
    }
  }

  // Inferred dimensions require a whole number of elements.
  auto invalid_size = allocations.front();
  --invalid_size.size_bytes;
  --invalid_size.reference_size_bytes;
  try {
    krepe::compare_views<int*>(invalid_size, [](auto, auto) {});
    return 1;
  } catch (const std::runtime_error&) {
  }

  return 0;
}
