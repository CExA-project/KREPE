#pragma once

#include "allocation_tracker.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace krepe {

struct ViewDumpResult {
  bool ok = false;
  std::string filename;
  std::string replay_filename;
  std::string error;
};

struct IndexTypeDesc {
  std::size_t size;
  bool is_signed;
};

struct NoPolicyDesc {};

struct ScalarPolicyDesc {
  std::uint64_t end;
};

struct RangePolicyDesc {
  IndexTypeDesc index_type_desc;
  const char* space;
  const char* schedule;
  std::uint64_t begin;
  std::uint64_t end;
  int chunk_size;
};

struct MDRangePolicyDesc {
  IndexTypeDesc index_type_desc;
  const char* space;
  const char* schedule;
  std::size_t rank;
  const char* outer_dir;
  const char* inner_dir;
  std::vector<std::int64_t> begin;
  std::vector<std::int64_t> end;
  std::vector<std::int64_t> tile;
};

struct TeamPolicyDesc {
  IndexTypeDesc index_type_desc;
  const char* space;
  const char* schedule;
  int team_size;
  int league_size;
  int vector_length;
  int team_scratch_0;
  int team_scratch_1;
  int thread_scratch_0;
  int thread_scratch_1;
  int chunk_size;
};

ViewDumpResult create_kernel_dump(
    const AllocationSnapshot& input_snapshot,
    const std::vector<unsigned char>& functor_data,
    const std::vector<unsigned char>& nvcc_inner_lambda_data,
    const std::unordered_map<std::string, std::string>& metadata,
    const std::variant<krepe::NoPolicyDesc, krepe::ScalarPolicyDesc,
                       krepe::RangePolicyDesc, krepe::MDRangePolicyDesc,
                       krepe::TeamPolicyDesc>& policy,
    const std::string& replay_source, std::string_view label,
    std::uint64_t kernel_id, std::uint64_t kernel_invocation);

ViewDumpResult append_kernel_output(const AllocationSnapshot& output_snapshot,
                                    std::string_view filename);

}  // namespace krepe
