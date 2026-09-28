// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <Kokkos_Macros.hpp>

// Extended lambdas on nvcc are implemented with structs storing a copy of the
// captured variables and a pointer to a heap allocated buffer containing a host
// only version of the original lambda object. In order to replay an extended
// lambda we need to copy both the captured variables (used for the device
// version) and the lambda buffer (used for the host version).
//
// The main challenge is that we cannot access this buffer or its size from an
// extended lambda object. For accessing the buffer, we assume that the pointer
// to the buffer will be the last data member of the extended lambda, and just
// compute it as `ptr = (&lambda + sizeof(lambda)) - sizeof(void*)`. For the
// size, we get the types that are captured from the signature of the wrapper
// struct and take the sizeof of an host lambda capturing exactly the same
// types.
//
// An alternative would have been to use a user-defined operator new, record
// all allocated pointers and sizes, and retrieve the size of the allocation
// corresponding to the buffer's pointer. The drawbacks are that we perform
// extra host allocations for the allocation table before the replayer's
// initialization, and this also prevent users from using their own operator
// new.
// Another alternative would have been to use the gcc/clang builtins
// `__builtin_[dynamic_]object_size()` which can give the size of an object
// based on a pointer to said object, but these builtins only work with
// optimizations enabled (in my testing, it only worked with -O3).
//
// Note that this is fragile as it relies on implementation details, but we
// found no other alternative to get the information we need out of an extended
// lambda.
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_COMPILER_NVCC)
#define KERNEL_REPLAYER_USE_NVCC_HDL_WORKAROUND
#endif

namespace krepe::hdl_utils {
// Before the host compilation phase, nvcc will try to instantiate the function
// with a regular lambda type, but we don't expect it to be called at runtime as
// the specialization will be used by the host compiler once cudafe++ generates
// the hdl wrappers.
template <class T>
std::size_t hdl_host_lambda_size(T) {
  std::abort();
}

// Function used to return the size of a host only lambda with the same captures
// as an extended lambda.
// We rely on the type of nvcc's __nv_hdl_wrapper_t here, we cannot spell out
// the type explicitly as it doesn't exist until the final host compilation
// step.
template <template <bool, bool, bool, class, class, class...> class T,
          bool IsMutable, bool HasFuncPtrConv, bool NeverThrows, typename Tag,
          typename Fun, typename... Fields>
std::size_t hdl_host_lambda_size(
    const T<IsMutable, HasFuncPtrConv, NeverThrows, Tag, Fun, Fields...>&) {
  using lambda_size_t = decltype([]<class... Args>(Args... args) {
    auto lambda = [=]() { ((void)args, ...); };
    return std::integral_constant<std::size_t, sizeof(lambda)>{};
  }((std::declval<Fields>())...));

  return lambda_size_t::value;
}

// Returns the pointer to the host lambda stored in extended lambdas.
// We assume that it is the last member of __nv_hdl_wrapper_t.
template <class Functor>
void* hdl_host_lambda_pointer(Functor&& f) {
  unsigned char* data = reinterpret_cast<unsigned char*>(&f);
  return *reinterpret_cast<void**>((data + sizeof(f)) - sizeof(void*));
}

template <class Functor>
constexpr bool lambda_is_hdl() {
#if defined(KOKKOS_COMPILER_NVCC)
  return __nv_is_extended_host_device_lambda_closure_type(
      std::remove_cvref_t<Functor>);
#else
  return false;
#endif
}

}  // namespace krepe::hdl_utils
