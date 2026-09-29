// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#pragma once

#include "allocation_tracker.hpp"

#include <string>
#include <vector>

namespace krepe {

std::string copy_allocation_bytes(const ActiveAllocation& allocation,
                                  std::vector<unsigned char>& bytes);

}  // namespace krepe
