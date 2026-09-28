// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include <string_view>

namespace krepe::impl {

enum class MemorySpaceType {
  HOST,
  DEVICE,
};

inline constexpr MemorySpaceType memory_space_type_from_string(
    std::string_view space_name) {
  return space_name == "Host" ? MemorySpaceType::HOST : MemorySpaceType::DEVICE;
}
}  // namespace krepe::impl
