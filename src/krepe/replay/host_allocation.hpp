// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#pragma once

#include <unistd.h>
#include <sys/mman.h>
#include <Kokkos_Core.hpp>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>

namespace krepe::impl {

inline auto regular_host_allocate(std::size_t size, char* data) {
  void* ptr = std::malloc(size);
  std::memcpy(ptr, data, size);
  return std::unique_ptr<void, void (*)(void*)>(ptr, std::free);
}

inline std::size_t host_allocation_granularity() { return getpagesize(); }

inline void host_copy_data(char* address, char* data, std::size_t size) {
  std::memcpy(address, data, size);
}

inline std::tuple<void*, std::size_t> host_allocate(void* aligned_address,
                                                    std::size_t size) {
  assert(reinterpret_cast<std::uintptr_t>(aligned_address) %
             host_allocation_granularity() ==
         0);

  void* address =
      mmap(aligned_address, size, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);

  if (address == MAP_FAILED) {
    if (errno == EEXIST) {
      // FIXME: better error message
      throw std::runtime_error(
          "Failed to allocate to a predefined host address");
    } else {
      throw std::runtime_error(
          "Unknown error when trying to allocate to a predefined host "
          "address: " +
          std::string(std::strerror(errno)));
    }
  } else if (address != aligned_address) {
    // We only reach this branch when the target address already belongs to a
    // reservation on older kernels which don't handle the MAP_FIXED_NOREPLACE
    // case. In that case the kernel allocates on another available address.
    munmap(address, size);
    throw std::runtime_error("Failed to allocate to a predefined host address");
  }

  return std::make_tuple(address, size);
}

inline void host_deallocate(void* address, std::size_t size) {
  munmap(address, size);
}

}  // namespace krepe::impl
