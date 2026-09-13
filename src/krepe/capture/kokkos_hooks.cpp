/*
 * Kokkos Tools hook for observing user-visible Kokkos activity.
 *
 * This file is compiled as a shared library and loaded at runtime through
 * KOKKOS_TOOLS_LIBS. Kokkos discovers the exported kokkosp_* symbols and calls
 * them when kernels start/end and when data is allocated or deallocated.
 *
 * The hook keeps the output focused on application code by filtering internal
 * Kokkos labels. For the selected kernel label, it dumps active Kokkos data
 * allocations to HDF5 before and after the selected kernel
 */

#include <impl/Kokkos_Profiling_C_Interface.h>

#include "allocation_tracker.hpp"
#include "view_dump.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <optional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <variant>
#include <vector>

#if defined(__linux__)
#include <malloc.h>
#endif

#if defined(KREPE_ENABLE_CUDA_DUMP)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#if defined(KREPE_ENABLE_HIP_DUMP)
#include <hip/hip_runtime_api.h>
#endif

#define KOKKOS_HOOKS_EXPORT __attribute__((visibility("default")))

#if !defined(KREPE_KOKKOS_ALLOCATION_HEADER_SIZE)
#error "KREPE_KOKKOS_ALLOCATION_HEADER_SIZE must be defined by CMake"
#endif

static_assert(KREPE_KOKKOS_ALLOCATION_HEADER_SIZE == 128 ||
                  KREPE_KOKKOS_ALLOCATION_HEADER_SIZE == 256,
              "unexpected Kokkos allocation header size");

namespace {

std::atomic<std::uint64_t> next_kernel_id{1};

std::mutex log_mutex;
std::mutex state_mutex;

struct KernelState {
  std::uint32_t device_id;
  std::uint64_t invocation;
  std::string dump_path;
};

std::unordered_map<std::uint64_t, KernelState> kernel_states;
std::unordered_map<std::string, std::uint64_t> kernel_invocations;
Kokkos_Tools_toolInvokedFenceFunction tool_fence = nullptr;

krepe::AllocationTracker allocation_tracker;
std::vector<unsigned char> functor_data;
std::vector<unsigned char> nvcc_inner_lambda_data;
std::unordered_map<std::string, std::string> metadata;
std::variant<krepe::NoPolicyDesc, krepe::ScalarPolicyDesc,
             krepe::RangePolicyDesc, krepe::MDRangePolicyDesc,
             krepe::TeamPolicyDesc>
    policy = krepe::NoPolicyDesc{};

const char* replay_source_fmt_string = nullptr;
std::string replay_source;

std::string dump_kernel_label;
std::optional<std::uint64_t> dump_kernel_invocation = 1;

constexpr std::string_view dump_kernel_label_option =
    "--krepe-dump-kernel-label=";
constexpr std::string_view dump_kernel_invocation_option =
    "--krepe-dump-kernel-invocation=";

std::string label_or_unknown(const char* label) {
  return label != nullptr ? label : "<unknown>";
}

bool is_internal_label(std::string_view label) {
  return label.starts_with("Kokkos::") || label.starts_with("HostSpace::") ||
         label.starts_with("CudaSpace::") || label.starts_with("HIPSpace::") ||
         label.starts_with("SYCLSpace::") ||
         label.starts_with("OpenACCSpace::") ||
         label.starts_with("NextSiliconSpace::") ||
         label.starts_with("KOKKOS_") || label.starts_with("kokkos.");
}

#if defined(KREPE_ENABLE_CUDA_DUMP)
bool is_cuda_pointer_attribute_space(const std::string& space) {
  return space == "Cuda" || space == "CudaUVM" || space == "CudaHostPinned";
}
#endif

#if defined(KREPE_ENABLE_HIP_DUMP)
bool is_hip_pointer_attribute_space(const std::string& space) {
  return space == "HIP" || space == "HIPManaged" || space == "HIPHostPinned";
}
#endif

bool is_host_space(const std::string& space) { return space == "Host"; }

std::string bounded_string(const char* value, std::size_t max_size) {
  if (value == nullptr) {
    return "<null>";
  }

  const char* end = std::find(value, value + max_size, '\0');
  return std::string(value, end);
}

std::string space_name(const Kokkos_Profiling_SpaceHandle& handle) {
  return bounded_string(handle.name, sizeof(handle.name));
}

template <typename... Args>
void log_line(Args&&... args) {
  std::lock_guard<std::mutex> lock(log_mutex);
  std::cerr << "[krepe-capture] ";
  (std::cerr << ... << args);
  std::cerr << '\n';
}

bool should_dump_views_for_label(std::string_view label) {
  return !dump_kernel_label.empty() && label == dump_kernel_label;
}

bool should_dump_views_for_invocation(std::string_view label,
                                      const std::uint64_t invocation) {
  return dump_kernel_invocation.has_value() &&
         should_dump_views_for_label(label) &&
         invocation == dump_kernel_invocation.value();
}

std::optional<std::uint64_t> parse_positive_uint64(std::string_view value) {
  std::uint64_t parsed    = 0;
  const auto* const begin = value.begin();
  const auto* const end   = value.end();
  const auto result       = std::from_chars(begin, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end || parsed == 0) {
    return std::nullopt;
  }
  return parsed;
}

void log_dump_result(const char* phase, const krepe::ViewDumpResult& result,
                     const krepe::AllocationSnapshot& snapshot,
                     const std::uint64_t kernel_id,
                     const std::uint64_t invocation) {
  if (result.ok) {
    log_line("dump_written phase=", phase, " path=\"", result.filename,
             "\" kernel_id=", kernel_id, " invocation=", invocation,
             " active_allocations=", snapshot.allocations.size(),
             " active_bytes=", snapshot.active_bytes);
    if (!result.replay_filename.empty()) {
      log_line("replay source written to ", result.replay_filename);
    }
  } else {
    log_line("dump_failed phase=", phase, " path=\"", result.filename,
             "\" kernel_id=", kernel_id, " invocation=", invocation,
             " error=\"", result.error, "\"");
  }
}

krepe::ViewDumpResult dump_input_views(const std::string& label,
                                       const std::uint64_t kernel_id,
                                       const std::uint64_t invocation) {
  const krepe::AllocationSnapshot snapshot = allocation_tracker.snapshot();
  krepe::ViewDumpResult result             = krepe::create_kernel_dump(
      snapshot, functor_data, nvcc_inner_lambda_data, metadata, policy,
      replay_source, label, kernel_id, invocation);
  log_dump_result("in", result, snapshot, kernel_id, invocation);
  return result;
}

void dump_output_views(const std::string& dump_path,
                       const std::uint64_t kernel_id,
                       const std::uint64_t invocation) {
  const krepe::AllocationSnapshot snapshot = allocation_tracker.snapshot();
  krepe::ViewDumpResult result =
      krepe::append_kernel_output(snapshot, dump_path);
  log_dump_result("out", result, snapshot, kernel_id, invocation);
}

bool should_track_allocation(const char* label, const void* ptr) {
  return ptr != nullptr && !is_internal_label(label_or_unknown(label));
}

const void* allocation_data_pointer(const void* ptr) {
  return static_cast<const unsigned char*>(ptr) +
         KREPE_KOKKOS_ALLOCATION_HEADER_SIZE;
}

std::size_t host_allocation_size(const void* ptr) {
#if defined(__linux__)
  // The allocator may reserve more memory than Kokkos requested, so
  // malloc_usable_size(ptr) can return a larger size than the requested one.
  return malloc_usable_size(const_cast<void*>(ptr));
#else
  (void)ptr;
  return 0;
#endif
}

std::optional<std::uint64_t> validated_data_size(
    const void* allocation_base, const std::size_t allocation_size,
    const void* data_ptr, const std::uint64_t reported_size) {
  const auto base = reinterpret_cast<std::uintptr_t>(allocation_base);
  const auto data = reinterpret_cast<std::uintptr_t>(data_ptr);
  if (data < base) {
    return std::nullopt;
  }

  const auto offset = data - base;
  if (offset > allocation_size) {
    return std::nullopt;
  }

  if (offset == allocation_size) {
    return 0;
  }

  const auto available = static_cast<std::uint64_t>(allocation_size - offset);
  return available >= reported_size ? reported_size : 0;
}

std::optional<std::uint64_t> allocation_data_size(
    const std::string& space, const void* ptr, const void* data_ptr,
    const std::uint64_t reported_size) {
  // Kokkos <= 5.2.0 reports the SharedAllocationHeader size instead of zero
  // for empty Views (kokkos/kokkos#9337). Disambiguate that legacy value using
  // the physical allocation bounds below.
  if (reported_size != KREPE_KOKKOS_ALLOCATION_HEADER_SIZE) {
    return reported_size;
  }

#if defined(KREPE_ENABLE_CUDA_DUMP)
  if (is_cuda_pointer_attribute_space(space)) {
    CUdeviceptr allocation_base       = 0;
    std::size_t allocation_size       = 0;
    const CUdeviceptr queried_pointer = reinterpret_cast<CUdeviceptr>(ptr);
    const CUresult base_error         = cuPointerGetAttribute(
        &allocation_base, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        queried_pointer);
    const CUresult size_error = cuPointerGetAttribute(
        &allocation_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE, queried_pointer);
    if (base_error == CUDA_SUCCESS && size_error == CUDA_SUCCESS) {
      return validated_data_size(reinterpret_cast<const void*>(allocation_base),
                                 allocation_size, data_ptr, reported_size);
    }
  }
#endif

#if defined(KREPE_ENABLE_HIP_DUMP)
  if (is_hip_pointer_attribute_space(space)) {
    hipDeviceptr_t allocation_base       = nullptr;
    std::size_t allocation_size          = 0;
    const hipDeviceptr_t queried_pointer = const_cast<void*>(ptr);
    const hipError_t base_error          = hipPointerGetAttribute(
        &allocation_base, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        queried_pointer);
    const hipError_t size_error = hipPointerGetAttribute(
        &allocation_size, HIP_POINTER_ATTRIBUTE_RANGE_SIZE, queried_pointer);
    if (base_error == hipSuccess && size_error == hipSuccess) {
      return validated_data_size(allocation_base, allocation_size, data_ptr,
                                 reported_size);
    }
  }
#endif

  if (is_host_space(space)) {
    const std::size_t allocation_size = host_allocation_size(ptr);
    if (allocation_size != 0) {
      return validated_data_size(ptr, allocation_size, data_ptr, reported_size);
    }
  }

  return std::nullopt;
}

void begin_kernel(const char* label, const std::uint32_t device_id,
                  std::uint64_t* kernel_id) {
  const std::uint64_t id         = next_kernel_id.fetch_add(1);
  const std::string kernel_label = label_or_unknown(label);
  std::uint64_t invocation       = 0;
  bool dump_this_kernel          = false;

  if (kernel_id != nullptr) {
    *kernel_id = id;
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex);
    invocation = ++kernel_invocations[kernel_label];
    dump_this_kernel =
        should_dump_views_for_invocation(kernel_label, invocation);
    kernel_states[id] = {device_id, invocation, {}};
  }

  if (dump_this_kernel) {
    log_line("kernel_selected label=\"", kernel_label,
             "\" invocation=", invocation, " kernel_id=", id);
    Kokkos_Tools_toolInvokedFenceFunction fence = nullptr;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      fence = tool_fence;
    }
    if (fence != nullptr) {
      fence(device_id);
    }
    const krepe::ViewDumpResult result =
        dump_input_views(kernel_label, id, invocation);
    std::lock_guard<std::mutex> lock(state_mutex);
    auto it = kernel_states.find(id);
    if (it != kernel_states.end() && result.ok) {
      it->second.dump_path = result.filename;
    }
  }
}

void end_kernel(const std::uint64_t kernel_id) {
  std::string dump_path;
  std::uint32_t device_id                     = 0;
  std::uint64_t invocation                    = 0;
  Kokkos_Tools_toolInvokedFenceFunction fence = nullptr;
  {
    std::lock_guard<std::mutex> lock(state_mutex);
    fence   = tool_fence;
    auto it = kernel_states.find(kernel_id);
    if (it != kernel_states.end()) {
      device_id  = it->second.device_id;
      invocation = it->second.invocation;
      dump_path  = it->second.dump_path;
      kernel_states.erase(it);
    }
  }

  if (!dump_path.empty()) {
    if (fence != nullptr) {
      fence(device_id);
    }
    dump_output_views(dump_path, kernel_id, invocation);
  }
}

void create_replay_source(const std::string& arg) {
  if (!replay_source_fmt_string) {
    return;
  }

  std::string_view fmt_string_view(replay_source_fmt_string);
  std::size_t placeholder_pos = fmt_string_view.find("{@}");

  if (placeholder_pos == std::string::npos) {
    log_line("[WARNING] Failed to find placeholder in the replay source");
    return;
  }

  replay_source = fmt_string_view.substr(0, placeholder_pos);
  replay_source += arg;
  replay_source += fmt_string_view.substr(placeholder_pos + 3);
}

}  // namespace

extern "C" {
KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_add_metadata(const char* key,
                                                        const char* value,
                                                        std::size_t size) {
  // We use the (ptr, size) constructor to accommodate strings containing null
  // characters
  metadata[key] = std::string(value, size);
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_copy_functor(
    const unsigned char* data, std::size_t size) {
  functor_data.resize(size);
  std::memcpy(functor_data.data(), data, size);
  nvcc_inner_lambda_data.clear();
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_copy_nvcc_lambda(
    const unsigned char* data, std::size_t size,
    const unsigned char* inner_functor_data, std::size_t inner_functor_size) {
  functor_data.resize(size);
  std::memcpy(functor_data.data(), data, size);
  nvcc_inner_lambda_data.resize(inner_functor_size);
  std::memcpy(nvcc_inner_lambda_data.data(), inner_functor_data,
              inner_functor_size);
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_register_scalar_policy(
    std::uint64_t N) {
  // FIXME: don't set the policy variant if the auto-extraction is enabled
  policy = krepe::ScalarPolicyDesc(N);

  if (!replay_source_fmt_string) {
    return;
  }

  create_replay_source(std::to_string(N));
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_register_range_policy(
    const char* space, const char* schedule, std::size_t index_type_size,
    bool index_type_signed, std::uint64_t begin, std::uint64_t end,
    int chunk_size) {
  policy = krepe::RangePolicyDesc({index_type_size, index_type_signed}, space,
                                  schedule, begin, end, chunk_size);

  if (!replay_source_fmt_string) {
    return;
  }

  std::ostringstream stream;
  stream << '(' << begin << ", " << end << ", Kokkos::ChunkSize(" << chunk_size
         << ')';
  create_replay_source(stream.str());
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_register_mdrange_policy(
    const char* space, const char* schedule, std::size_t rank,
    const char* outer_dir, const char* inner_dir, std::size_t index_type_size,
    bool index_type_signed, const std::int64_t* begin, const std::int64_t* end,
    const std::int64_t* tile) {
  policy = krepe::MDRangePolicyDesc(
      {index_type_size, index_type_signed}, space, schedule, rank, outer_dir,
      inner_dir, std::vector(begin, begin + rank), std::vector(end, end + rank),
      std::vector(tile, tile + rank));

  if (!replay_source_fmt_string) {
    return;
  }

  std::ostringstream stream;
  stream << "({" << begin[0];
  for (std::size_t i = 1; i < rank; i++) {
    stream << ", " << begin[i];
  }
  stream << "}, {" << end[0];
  for (std::size_t i = 1; i < rank; i++) {
    stream << ", " << end[i];
  }
  stream << "}, {" << tile[0];
  for (std::size_t i = 1; i < rank; i++) {
    stream << ", " << tile[i];
  }
  stream << "})";
  create_replay_source(stream.str());
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_register_team_policy(
    const char* space, const char* schedule, std::size_t index_type_size,
    bool index_type_signed, int team_size, int league_size, int vector_length,
    int team_scratch_0, int team_scratch_1, int thread_scratch_0,
    int thread_scratch_1, int chunk_size) {
  policy = krepe::TeamPolicyDesc(
      {index_type_size, index_type_signed}, space, schedule, team_size,
      league_size, vector_length, team_scratch_0, team_scratch_1,
      thread_scratch_0, thread_scratch_1, chunk_size);

  if (!replay_source_fmt_string) {
    return;
  }

  std::ostringstream stream;
  stream << '(' << league_size << ", ";
  if (team_size > 0) {
    stream << team_size;
  } else {
    stream << "Kokkos::AUTO";
  }
  stream << ", ";
  if (vector_length > 0) {
    stream << vector_length;
  } else {
    stream << "Kokkos::AUTO";
  }
  stream << ").set_chunk_size(" << chunk_size << ')';
  if (team_scratch_0 >= 0) {
    stream << ".set_scratch_size(0, Kokkos::PerTeam(" << team_scratch_0
           << "), Kokkos::PerThread(" << thread_scratch_0 << ")";
    stream << ".set_scratch_size(1, Kokkos::PerTeam(" << team_scratch_1
           << "), Kokkos::PerThread(" << thread_scratch_1 << ")";
  }
  create_replay_source(stream.str());
}

KOKKOS_HOOKS_EXPORT bool krepe_kernel_dump_next_invocation_will_dump(
    const char* kernel_name) {
  std::lock_guard<std::mutex> lock(state_mutex);
  return should_dump_views_for_invocation(kernel_name,
                                          kernel_invocations[kernel_name] + 1);
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_clear_registered_views() {
  allocation_tracker.clear_used_allocations();
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_register_view(void* data,
                                                         const char* space) {
  allocation_tracker.record_used_allocation(space, data);
}

KOKKOS_HOOKS_EXPORT void krepe_kernel_dump_register_replay_source(
    const char* source) {
  replay_source_fmt_string = source;
}

KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_for(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel(label, device_id, kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_for(
    const std::uint64_t kernel_id) {
  end_kernel(kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_reduce(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel(label, device_id, kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_reduce(
    const std::uint64_t kernel_id) {
  end_kernel(kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_begin_parallel_scan(
    const char* label, const std::uint32_t device_id,
    std::uint64_t* kernel_id) {
  begin_kernel(label, device_id, kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_end_parallel_scan(
    const std::uint64_t kernel_id) {
  end_kernel(kernel_id);
}

KOKKOS_HOOKS_EXPORT void kokkosp_parse_args(const int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view argument = argv[i] != nullptr ? argv[i] : "";
    if (argument.starts_with(dump_kernel_label_option)) {
      dump_kernel_label =
          std::string(argument.substr(dump_kernel_label_option.size()));
    } else if (argument.starts_with(dump_kernel_invocation_option)) {
      const std::string_view value =
          argument.substr(dump_kernel_invocation_option.size());
      dump_kernel_invocation = parse_positive_uint64(value);
      if (!dump_kernel_invocation.has_value()) {
        log_line("invalid ", dump_kernel_invocation_option, " value=\"", value,
                 "\"; expected a positive integer");
      }
    }
  }
}

KOKKOS_HOOKS_EXPORT void kokkosp_allocate_data(
    const Kokkos_Profiling_SpaceHandle space, const char* label,
    const void* ptr, const std::uint64_t size) {
  const std::string allocation_label = label_or_unknown(label);
  const std::string allocation_space = space_name(space);

  if (should_track_allocation(label, ptr)) {
    const void* p_data = allocation_data_pointer(ptr);
    const std::optional<std::uint64_t> data_size =
        allocation_data_size(allocation_space, ptr, p_data, size);
    const std::uint64_t tracked_size = data_size.value_or(0);

    allocation_tracker.record_allocation(allocation_label, allocation_space,
                                         ptr, p_data, tracked_size, size,
                                         data_size.has_value());
  }
}

KOKKOS_HOOKS_EXPORT void kokkosp_deallocate_data(
    const Kokkos_Profiling_SpaceHandle space, const char*, const void* ptr,
    const std::uint64_t) {
  const std::string deallocation_space = space_name(space);

  if (ptr != nullptr) {
    allocation_tracker.record_deallocation(deallocation_space, ptr);
  }
}

KOKKOS_HOOKS_EXPORT void kokkosp_provide_tool_programming_interface(
    const std::uint32_t, Kokkos_Tools_ToolProgrammingInterface interface) {
  std::lock_guard<std::mutex> lock(state_mutex);
  tool_fence = interface.fence;
}

KOKKOS_HOOKS_EXPORT void kokkosp_request_tool_settings(
    const std::uint32_t, Kokkos_Tools_ToolSettings* settings) {
  if (settings != nullptr) {
    *settings                         = {};
    settings->requires_global_fencing = false;
  }
}
}
