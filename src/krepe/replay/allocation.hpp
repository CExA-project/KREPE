// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#pragma once

#include <string>
#include <tuple>

#include "memory_space_type.hpp"
#include "host_allocation.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#define KERNEL_REPLAYER_HAS_DEVICE_SPACE
#include "cuda_allocation.hpp"
#elif defined(KOKKOS_ENABLE_HIP)
#define KERNEL_REPLAYER_HAS_DEVICE_SPACE
#include "hip_allocation.hpp"
#elif defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENACC) || \
    defined(KOKKOS_ENABLE_NEXTSILICON)
#define KERNEL_REPLAYER_HAS_UNSUPPORTED_DEVICE_SPACE
#endif

namespace krepe::impl {

inline std::pair<char*, std::size_t> get_allocation_address(
    void* target_address, std::size_t size, MemorySpaceType space) {
  std::size_t granularity;
  if (space == MemorySpaceType::HOST) {
    granularity = host_allocation_granularity();
  } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    granularity = device_allocation_granularity();
#else
    throw std::runtime_error(
        "Tried to compute a device allocation address but no device space is "
        "enabled");
#endif
  }

  char* aligned_address = reinterpret_cast<char*>(
      ((reinterpret_cast<std::uintptr_t>(target_address)) / granularity) *
      granularity);
  size += reinterpret_cast<char*>(target_address) - aligned_address;

  std::size_t total_size = (((size - 1) / granularity) + 1) * granularity;

  return {aligned_address, total_size};
}

inline void copy_data(MemorySpaceType space, char* address, char* data,
                      std::size_t size) {
  if (space == MemorySpaceType::HOST) {
    host_copy_data(address, data, size);
  } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    device_copy_data(address, data, size);
#else
    throw std::runtime_error(
        "Tried to copy data to a device space but no device backend is "
        "enabled");
#endif
  }
}

struct Allocation {
  void* address    = nullptr;
  std::size_t size = 0;
  MemorySpaceType memory_space;

  Allocation() = default;

  Allocation(MemorySpaceType memory_space, char* target_address,
             std::size_t requested_size)
      : memory_space(memory_space) {
    if (memory_space == MemorySpaceType::HOST) {
      std::tie(address, size) = host_allocate(target_address, requested_size);
    } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
      std::tie(address, size) = device_allocate(target_address, requested_size);
#elif defined(KERNEL_REPLAYER_HAS_UNSUPPORTED_DEVICE_SPACE)
      throw std::runtime_error(
          "Trying to allocate on an unsupported device space");
#else
      throw std::runtime_error(
          "Trying to allocate on device but no device backend has been "
          "enabled");
#endif
    }
  }

  ~Allocation() {
    if (address == nullptr) {
      return;
    }

    if (memory_space == MemorySpaceType::HOST) {
      host_deallocate(address, size);
    } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
      device_deallocate(address, size);
#endif
    }
  }

  Allocation(const Allocation&)            = delete;
  Allocation& operator=(const Allocation&) = delete;

  Allocation(Allocation&& other) : address(other.address), size(other.size) {
    other.address = nullptr;
    other.size    = 0;
  }

  Allocation& operator=(Allocation&& other) {
    address = other.address;
    size    = other.size;

    other.address = nullptr;
    other.size    = 0;

    return *this;
  }
};
}  // namespace krepe::impl
