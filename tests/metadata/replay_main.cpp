// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>
#include <sstream>
#include <stdexcept>

template <class T, class U>
void krepe_expect(const T& a, const U& b, const char* expr_a,
                  const char* expr_b) {
  if (a != b) {
    std::ostringstream os;
    os << "Assertion " << expr_a << " == " << expr_b << " failed, got " << a
       << " but expected " << b << '\n';
    throw std::runtime_error(os.str());
  }
}

#define KREPE_EXPECT(a, b) krepe_expect((a), (b), #a, #b)

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  KREPE_EXPECT(krepe::get_metadata("foo").value(), "bar");
  KREPE_EXPECT(krepe::get_metadata("with null bytes").value(),
               std::string("hello\0world", 12));
  KREPE_EXPECT(krepe::get_metadata("n_iter").value(), "10");
  KREPE_EXPECT(krepe::get_metadata("missing").has_value(), false);

  int n_iter = std::stoi(krepe::get_metadata("n_iter").value());

  int sum = 0;
  Kokkos::parallel_reduce(
      "test_kernel", n_iter,
      krepe::replay_functor(KOKKOS_LAMBDA(int, int& sum) { sum++; }), sum);

  Kokkos::printf("Result is %d\n", sum);

  return 0;
}
