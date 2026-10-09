// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#pragma once

#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <Kokkos_Macros.hpp>

// Extended lambdas on nvcc are implemented with structs storing a copy of the
// captured variables and a pointer to a heap allocated buffer containing a host
// only version of the original lambda object. In order to replay an extended
// lambda we need to copy both the captured variables (used for the device
// version) and the lambda buffer (used for the host version).
//
// Captured extended lambdas have their own host closures in both the wrapper
// and the host closure. These pointers belong to the capturing process and
// must be replaced during replay, so both sets of captures are visited.
//
// The main challenge is that we cannot access this buffer or its size through
// a public interface of an extended lambda object. For accessing the buffer,
// we use the data member of the wrapper struct during the host compilation
// phase. We cannot compute its position as sizeof(lambda) - sizeof(void*)
// because over-aligned captures can add padding after the pointer. For the
// size, we get the types that are captured from the signature of the wrapper
// struct and compute their layout using sizeof and alignof of each type.
// This assumes that the wrapper and host lambda store their captures in the
// same order and with the same alignment.
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

template <class Functor>
constexpr bool lambda_is_hdl() {
#if defined(KOKKOS_COMPILER_NVCC)
  return __nv_is_extended_host_device_lambda_closure_type(
      std::remove_cvref_t<Functor>);
#else
  return false;
#endif
}

constexpr std::size_t align_offset(std::size_t offset, std::size_t alignment) {
  return (offset + alignment - 1) / alignment * alignment;
}

template <class... Fields>
constexpr std::size_t host_capture_size() {
  if constexpr (sizeof...(Fields) == 0) {
    return 1;
  } else {
    std::size_t size      = 0;
    std::size_t alignment = 1;

    const auto add_capture = [&](std::size_t capture_size,
                                 std::size_t capture_alignment) {
      size = align_offset(size, capture_alignment);
      size += capture_size;
      if (capture_alignment > alignment) {
        alignment = capture_alignment;
      }
    };
    (add_capture(sizeof(Fields), alignof(Fields)), ...);

    return align_offset(size, alignment);
  }
}

// Before the host compilation phase, nvcc will try to instantiate the function
// with a regular lambda type, but we don't expect it to be called at runtime as
// the specialization will be used by the host compiler once cudafe++ generates
// the hdl wrappers.
template <class T, class Visitor>
void visit_hdl_host_lambdas(const T&, Visitor&) {
  std::abort();
}

// We rely on the type of nvcc's __nv_hdl_wrapper_t here, we cannot spell out
// the type explicitly as it doesn't exist until the final host compilation
// step.
template <template <bool, bool, bool, class, class, class...> class T,
          bool IsMutable, bool HasFuncPtrConv, bool NeverThrows, class Tag,
          class Fun, class... Fields, class Visitor>
void visit_hdl_host_lambdas(
    const T<IsMutable, HasFuncPtrConv, NeverThrows, Tag, Fun, Fields...>& f,
    Visitor& visitor);

template <class... Fields, class Visitor>
void visit_hdl_captures(const void* storage, Visitor& visitor) {
  if constexpr (sizeof...(Fields) > 0) {
    std::size_t offset     = 0;
    const auto* bytes      = static_cast<const unsigned char*>(storage);
    const auto visit_field = [&]<class Field>() {
      offset = align_offset(offset, alignof(Field));
      if constexpr (lambda_is_hdl<Field>()) {
        visit_hdl_host_lambdas(*reinterpret_cast<const Field*>(bytes + offset),
                               visitor);
      }
      offset += sizeof(Field);
    };
    (visit_field.template operator()<Fields>(), ...);
  }
}

template <template <bool, bool, bool, class, class, class...> class T,
          bool IsMutable, bool HasFuncPtrConv, bool NeverThrows, class Tag,
          class Fun, class... Fields, class Visitor>
void visit_hdl_host_lambdas(
    const T<IsMutable, HasFuncPtrConv, NeverThrows, Tag, Fun, Fields...>& f,
    Visitor& visitor) {
  static_assert(!HasFuncPtrConv,
                "NVCC lambdas without a host lambda buffer are not supported");
  static_assert(host_capture_size<Fields...>() <= sizeof(f));
  // Access the member directly: over-aligned captures can add padding after
  // this pointer, so it is not necessarily at sizeof(f) - sizeof(void*).

  // Visit the parent before its children: restoring its host buffer also
  // replaces the bytes of the wrappers captured inside that buffer.
  visitor(&f, const_cast<void**>(&f.data), f.data,
          host_capture_size<Fields...>());
  visit_hdl_captures<Fields...>(&f, visitor);
  visit_hdl_captures<Fields...>(f.data, visitor);
}

}  // namespace krepe::hdl_utils
