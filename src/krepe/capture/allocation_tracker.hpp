// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace krepe {

struct AllocationRecord {
  std::string label;
  std::string space;
  const void* p_data;
  std::uint64_t size;
  std::uint64_t reported_size;
  bool data_size_known;
};

struct ActiveAllocation {
  const void* ptr;
  AllocationRecord record;
};

struct AllocationSnapshot {
  std::uint64_t active_bytes = 0;
  std::vector<ActiveAllocation> allocations;
};

class AllocationTracker {
 public:
  void record_allocation(std::string label, std::string space, const void* ptr,
                         const void* p_data, std::uint64_t size,
                         std::uint64_t reported_size, bool data_size_known);
  void record_used_allocation(const char* space, const void* ptr);
  void clear_used_allocations();
  void record_deallocation(std::string space, const void* ptr);

  AllocationSnapshot snapshot();

 private:
  mutable std::mutex mutex_;

  std::unordered_map<std::string, std::vector<const void*>> used_allocations_;

  std::unordered_map<std::string,
                     std::unordered_map<const void*, AllocationRecord>>
      active_allocations_;
};

}  // namespace krepe
