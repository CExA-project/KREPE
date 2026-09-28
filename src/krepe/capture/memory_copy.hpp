// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include "allocation_tracker.hpp"

#include <string>
#include <vector>

namespace krepe {

std::string copy_allocation_bytes(const ActiveAllocation& allocation,
                                  std::vector<unsigned char>& bytes);

}  // namespace krepe
