// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

#include <stdexcept>
#include <string_view>
#include <tuple>

struct MultiplyFunctor {
  int factor;
  Kokkos::View<int*> values;

  KOKKOS_FUNCTION void operator()(const int i) const { values(i) *= factor; }
};

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values;
  krepe::parallel_for("test_kernel", 0, MultiplyFunctor{0, values});
  Kokkos::fence();

  try {
    krepe::compare_views<int*>("values", std::make_tuple(N), [](auto, auto) {});
  } catch (const std::runtime_error& error) {
    constexpr std::string_view expected_error =
        "Reference output for view 'values' is not available";
    const std::string_view message = error.what();
    return message.find(expected_error) != std::string_view::npos ? 0 : 1;
  }

  return 1;
}
