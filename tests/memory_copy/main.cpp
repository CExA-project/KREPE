// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include "memory_copy.hpp"

#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace {

int expect_known_empty_allocation_to_succeed(const std::string& space) {
  const unsigned char allocation_identity = 0;
  const krepe::ActiveAllocation allocation{
      &allocation_identity,
      {"empty", space, nullptr, 0, 0, true},
  };
  std::vector<unsigned char> bytes{0xff};

  const std::string reason = krepe::copy_allocation_bytes(allocation, bytes);
  if (!reason.empty()) {
    std::cerr << space << ": expected a successful zero-byte copy, got \""
              << reason << "\"\n";
    return 1;
  }
  if (!bytes.empty()) {
    std::cerr << space << ": expected an empty byte buffer\n";
    return 1;
  }

  return 0;
}

}  // namespace

int main() {
  int failures = 0;
  for (const std::string& space :
       std::array<std::string, 5>{"SYCLDeviceUSM", "OpenACCSpace",
                                  "UnsupportedSpace", "Cuda", "HIP"}) {
    failures += expect_known_empty_allocation_to_succeed(space);
  }

  const unsigned char allocation_identity = 0;
  const krepe::ActiveAllocation unknown_size_allocation{
      &allocation_identity,
      {"unknown", "UnsupportedSpace", nullptr, 0, 0, false},
  };
  std::vector<unsigned char> bytes;
  const std::string reason =
      krepe::copy_allocation_bytes(unknown_size_allocation, bytes);
  if (reason != "allocation data size could not be bounded safely") {
    std::cerr << "expected an unknown size to be rejected before memory-space "
                 "validation, got \""
              << reason << "\"\n";
    ++failures;
  }

  return failures == 0 ? 0 : 1;
}
