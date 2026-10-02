// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#include "allocation_tracker.hpp"

#include <algorithm>
#include <utility>

namespace krepe {

void AllocationTracker::record_allocation(std::string label, std::string space,
                                          const void* ptr, const void* p_data,
                                          const std::uint64_t size,
                                          const std::uint64_t reported_size,
                                          const bool data_size_known) {
  if (ptr == nullptr) {
    return;
  }

  AllocationRecord record{std::move(label), space,          p_data, size,
                          reported_size,    data_size_known};
  std::lock_guard<std::mutex> lock(mutex_);
  active_allocations_[space].insert_or_assign(ptr, std::move(record));
}

void AllocationTracker::record_deallocation(std::string space,
                                            const void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto space_it = active_allocations_.find(space);
  if (space_it == active_allocations_.end()) {
    return;
  }

  space_it->second.erase(ptr);
  if (space_it->second.empty()) {
    active_allocations_.erase(space_it);
  }
}

void AllocationTracker::record_used_allocation(const char* space,
                                               const void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  used_allocations_[space].push_back(ptr);
}

void AllocationTracker::clear_used_allocations() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [space, pointers] : used_allocations_) {
    pointers.clear();
  }
}

AllocationSnapshot AllocationTracker::snapshot() {
  std::lock_guard<std::mutex> lock(mutex_);
  AllocationSnapshot snapshot;
#if defined(KREPE_HAS_COMPILER_PLUGINS)
  for (auto& [space, pointers] : used_allocations_) {
    std::sort(pointers.begin(), pointers.end());

    for (const auto& [ptr, record] : active_allocations_[space]) {
      auto used_ptr =
          std::lower_bound(pointers.begin(), pointers.end(), record.p_data);
      if (used_ptr != pointers.end() &&
          reinterpret_cast<std::uintptr_t>(*used_ptr) <=
              reinterpret_cast<std::uintptr_t>(record.p_data) + record.size) {
        snapshot.active_bytes += record.size;
        snapshot.allocations.push_back({ptr, record});
      }
    }
  }
#else
  for (const auto& [space, allocations] : active_allocations_) {
    for (const auto& [ptr, record] : allocations) {
      snapshot.active_bytes += record.size;
      snapshot.allocations.push_back({ptr, record});
    }
  }
#endif

  return snapshot;
}

}  // namespace krepe
