// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#pragma once

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <cuda.h>
#include <Kokkos_Core.hpp>

namespace krepe::impl {
inline void throw_error(CUresult error, const char* expr, const char* file,
                        int line) {
  if (error == CUDA_SUCCESS) {
    return;
  }
  const char* name        = nullptr;
  const char* description = nullptr;
  cuGetErrorName(error, &name);
  cuGetErrorString(error, &description);
  std::ostringstream os;
  if (file) {
    os << file << ":" << line << ": ";
  }
  os << "Call " << expr << " failed with:\n";
  if (name) {
    os << name;
    if (description) {
      os << ": " << description;
    }
  }
  throw std::runtime_error(os.str());
}

#define CHECK_CUDA_CALL(expr) \
  ::krepe::impl::throw_error((expr), #expr, __FILE__, __LINE__)

inline auto regular_device_allocate(std::size_t size, char* data) {
  void* ptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&ptr, size));
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(ptr, data, size, cudaMemcpyHostToDevice));
  return std::unique_ptr<void, void (*)(void*)>(
      ptr, [](void* ptr) { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(ptr)); });
}

inline void device_copy_data(char* address, char* data, std::size_t size) {
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaMemcpy(address, data, size, cudaMemcpyHostToDevice));
}

inline std::size_t device_allocation_granularity() {
  CUdevice device;
  CHECK_CUDA_CALL(cuCtxGetDevice(&device));

  CUmemAllocationHandleType handleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUmemAllocationProp prop  = {};
  prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id          = device;
  prop.requestedHandleTypes = handleType;

  size_t granularity = 0;
  CHECK_CUDA_CALL(cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  // FIXME: investigate, on jean-zay the allocated addresses are multiple of
  // this, even though granularity is way smaller
  granularity = 0x2000000;

  return granularity;
}

inline std::tuple<void*, std::size_t> device_allocate(void* address,
                                                      std::size_t size) {
  CUdevice device;
  CHECK_CUDA_CALL(cuCtxGetDevice(&device));

  // Allocate physical memory
  CUmemAllocationHandleType handleType =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUmemAllocationProp prop  = {};
  prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id          = device;
  prop.requestedHandleTypes = handleType;

  CUmemGenericAllocationHandle allocHandle;
  CHECK_CUDA_CALL(cuMemCreate(&allocHandle, size, &prop, 0));

  CUdeviceptr real_address   = 0;
  CUdeviceptr target_address = reinterpret_cast<CUdeviceptr>(address);
  // TODO: check if the alignment parameter should be equal to the granularity
  CHECK_CUDA_CALL(
      cuMemAddressReserve(&real_address, size, 64, target_address, 0));
  if (real_address != target_address) {
    throw std::runtime_error(
        "Failed to allocate data at a specific device address");
  }
  CHECK_CUDA_CALL(cuMemMap(real_address, size, 0, allocHandle, 0));

  // Make the address range accessible on device
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id     = device;
  accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CHECK_CUDA_CALL(cuMemSetAccess(real_address, size, &accessDesc, 1));

  CHECK_CUDA_CALL(cuMemRelease(allocHandle));

  return std::make_tuple(reinterpret_cast<void*>(real_address), size);
}

inline void device_deallocate(void* address, std::size_t size) {
  CHECK_CUDA_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(address), size));
  CHECK_CUDA_CALL(
      cuMemAddressFree(reinterpret_cast<CUdeviceptr>(address), size));
}
}  // namespace krepe::impl
