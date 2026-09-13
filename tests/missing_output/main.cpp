#include "view_dump.hpp"

#include <Kokkos_Core.hpp>

#include <cstring>
#include <vector>

struct MultiplyFunctor {
  int factor;
  Kokkos::View<int*> values;

  KOKKOS_FUNCTION void operator()(const int i) const { values(i) *= factor; }
};

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });
  Kokkos::fence();

  const std::uint64_t data_size = values.span() * sizeof(int);
  const krepe::AllocationSnapshot input_snapshot{
      data_size,
      {{values.data(),
        {"values", Kokkos::View<int*>::memory_space::name(), values.data(),
         data_size, data_size, true}}},
  };

  const MultiplyFunctor functor{2, values};
  std::vector<unsigned char> functor_data(sizeof(functor));
  std::memcpy(functor_data.data(), &functor, sizeof(functor));

  const auto dump = krepe::create_kernel_dump(input_snapshot, functor_data, {},
                                              {}, krepe::ScalarPolicyDesc{N},
                                              {}, "test_kernel", 1, 1);
  if (!dump.ok) {
    return 1;
  }

  return 0;
}
