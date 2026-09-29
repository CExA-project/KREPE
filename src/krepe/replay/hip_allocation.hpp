// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#pragma once

#include <cstring>
#include <tuple>
#include <utility>
#include <hip/hip_runtime.h>
#include <Kokkos_Core.hpp>

#define CHECK_HIP_CALL(expr) KOKKOS_IMPL_HIP_SAFE_CALL(expr)

namespace krepe::impl {
inline auto regular_device_allocate(std::size_t size, char* data) {
  void* ptr;
  CHECK_HIP_CALL(hipMalloc(&ptr, size));
  CHECK_HIP_CALL(hipMemcpy(ptr, data, size, hipMemcpyHostToDevice));
  return std::unique_ptr<void, void (*)(void*)>(
      ptr, [](void* ptr) { CHECK_HIP_CALL(hipFree(ptr)); });
}

inline void device_copy_data(char* address, char* data, std::size_t size) {
  CHECK_HIP_CALL(hipMemcpy(address, data, size, hipMemcpyHostToDevice));
}

inline std::size_t device_allocation_granularity() {
  int device;
  CHECK_HIP_CALL(hipGetDevice(&device));

  hipMemAllocationProp prop = {};
  prop.type                 = hipMemAllocationTypePinned;
  prop.location.type        = hipMemLocationTypeDevice;
  prop.location.id          = device;

  size_t granularity = 0;
  CHECK_HIP_CALL(hipMemGetAllocationGranularity(
      &granularity, &prop, hipMemAllocationGranularityMinimum));

  return granularity;
}

inline std::tuple<void*, std::size_t> device_allocate(void* target_address,
                                                      std::size_t size) {
  int device;
  CHECK_HIP_CALL(hipGetDevice(&device));

  // Allocate physical memory
  hipMemAllocationProp prop = {};
  prop.type                 = hipMemAllocationTypePinned;
  prop.location.type        = hipMemLocationTypeDevice;
  prop.location.id          = device;

  hipMemGenericAllocationHandle_t allocHandle;
  CHECK_HIP_CALL(hipMemCreate(&allocHandle, size, &prop, 0));

  void* real_address = nullptr;
  CHECK_HIP_CALL(
      hipMemAddressReserve(&real_address, size, 64, target_address, 0));
  CHECK_HIP_CALL(hipMemMap(real_address, size, 0, allocHandle, 0));

  // Make the address range accessible on device
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type    = hipMemLocationTypeDevice;
  accessDesc.location.id      = device;
  accessDesc.flags            = hipMemAccessFlagsProtReadWrite;
  CHECK_HIP_CALL(hipMemSetAccess(real_address, size, &accessDesc, 1));

  CHECK_HIP_CALL(hipMemRelease(allocHandle));

  return std::make_tuple(real_address, size);
}

inline void device_deallocate(void* address, std::size_t size) {
  CHECK_HIP_CALL(hipMemUnmap(address, size));
  CHECK_HIP_CALL(hipMemAddressFree(address, size));
}
}  // namespace krepe::impl
