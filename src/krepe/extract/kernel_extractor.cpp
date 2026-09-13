#include "kernel_extractor.hpp"
#include <iostream>
#if defined(KOKKOS_ENABLE_LIBDL)
#include <dlfcn.h>
#endif

#include <optional>

namespace krepe {
namespace impl {

void (*add_metadata_function)(const char*, const char*, std::size_t) = nullptr;
void (*copy_functor_function)(const unsigned char*, std::size_t)     = nullptr;
void (*copy_nvcc_lambda_function)(const unsigned char*, std::size_t,
                                  const unsigned char*, std::size_t) = nullptr;
void (*register_scalar_policy_function)(std::uint64_t)               = nullptr;
void (*register_range_policy_function)(const char*, const char*, std::size_t,
                                       bool, std::uint64_t, std::uint64_t,
                                       int)                          = nullptr;
void (*register_mdrange_policy_function)(const char*, const char*, std::size_t,
                                         const char*, const char*, std::size_t,
                                         bool, const std::int64_t*,
                                         const std::int64_t*,
                                         const std::int64_t*)        = nullptr;
void (*register_team_policy_function)(const char*, const char*, std::size_t,
                                      bool, int, int, int, int, int, int, int,
                                      int)                           = nullptr;
bool (*next_invocation_will_dump_function)(const char*)              = nullptr;
void (*clear_registered_views_function)()                            = nullptr;
void (*register_view_function)(void* data, const char*)              = nullptr;
void (*register_replay_source_function)(const char*)                 = nullptr;
std::optional<bool> has_kernel_dump_tool = std::nullopt;

#if defined(KOKKOS_ENABLE_LIBDL)
template <class FuncPtr>
bool load_function_from_tool(const char* name, FuncPtr& ptr) {
  void* handle = dlsym(RTLD_DEFAULT, name);
  if (dlerror() != nullptr) {
    return false;
  }

  ptr = reinterpret_cast<FuncPtr>(handle);
  return true;
}

bool init_internal_functions_from_tool() {
  dlerror();

  // TODO: print a warning when some (at least 1 but not all) functions cannot
  // be loaded to indicate that there is an issue
  return load_function_from_tool("krepe_kernel_dump_add_metadata",
                                 add_metadata_function) &&
         load_function_from_tool("krepe_kernel_dump_copy_functor",
                                 copy_functor_function) &&
         load_function_from_tool("krepe_kernel_dump_copy_nvcc_lambda",
                                 copy_nvcc_lambda_function) &&
         load_function_from_tool("krepe_kernel_dump_register_scalar_policy",
                                 register_scalar_policy_function) &&
         load_function_from_tool("krepe_kernel_dump_register_range_policy",
                                 register_range_policy_function) &&
         load_function_from_tool("krepe_kernel_dump_register_mdrange_policy",
                                 register_mdrange_policy_function) &&
         load_function_from_tool("krepe_kernel_dump_register_team_policy",
                                 register_team_policy_function) &&
         load_function_from_tool("krepe_kernel_dump_next_invocation_will_dump",
                                 next_invocation_will_dump_function) &&
         load_function_from_tool("krepe_kernel_dump_clear_registered_views",
                                 clear_registered_views_function) &&
         load_function_from_tool("krepe_kernel_dump_register_view",
                                 register_view_function) &&
         load_function_from_tool("krepe_kernel_dump_register_replay_source",
                                 register_replay_source_function);
}
#else
bool init_internal_functions_from_tool() { return false; }
#endif

void init_internal_functions() {
  if (!has_kernel_dump_tool.has_value()) {
    if (init_internal_functions_from_tool()) {
      has_kernel_dump_tool = true;
    } else {
      add_metadata_function     = [](const char*, const char*, std::size_t) {};
      copy_functor_function     = [](const unsigned char*, std::size_t) {};
      copy_nvcc_lambda_function = [](const unsigned char*, std::size_t,
                                     const unsigned char*, std::size_t) {};
      register_scalar_policy_function = [](std::uint64_t) {};
      register_range_policy_function = [](const char*, const char*, std::size_t,
                                          bool, std::uint64_t, std::uint64_t,
                                          int) {};
      register_mdrange_policy_function =
          [](const char*, const char*, std::size_t, const char*, const char*,
             std::size_t, bool, const std::int64_t*, const std::int64_t*,
             const std::int64_t*) {};
      register_team_policy_function = [](const char*, const char*, std::size_t,
                                         bool, int, int, int, int, int, int,
                                         int, int) {};
      next_invocation_will_dump_function = [](const char*) { return false; };
      clear_registered_views_function    = []() {};
      register_view_function             = [](void*, const char*) {};
      register_replay_source_function    = [](const char*) {};
      has_kernel_dump_tool               = false;
    }
  }
}

void copy_functor(const unsigned char* functor_data, std::size_t functor_size) {
  init_internal_functions();
  copy_functor_function(functor_data, functor_size);
}

void copy_functor(const unsigned char* functor_data, std::size_t functor_size,
                  const unsigned char* inner_functor_data,
                  std::size_t inner_functor_size) {
  init_internal_functions();
  copy_nvcc_lambda_function(functor_data, functor_size, inner_functor_data,
                            inner_functor_size);
}

void register_scalar_policy(std::uint64_t N) {
  init_internal_functions();
  register_scalar_policy_function(N);
}

void register_range_policy(const char* space, const char* schedule,
                           std::size_t index_type_size, bool index_type_signed,
                           std::uint64_t begin, std::uint64_t end,
                           int chunk_size) {
  init_internal_functions();
  register_range_policy_function(space, schedule, index_type_size,
                                 index_type_signed, begin, end, chunk_size);
}

void register_mdrange_policy(const char* space, const char* schedule,
                             std::size_t rank, const char* outer_dir,
                             const char* inner_dir, std::size_t index_type_size,
                             bool index_type_signed, const std::int64_t* begin,
                             const std::int64_t* end,
                             const std::int64_t* tile) {
  init_internal_functions();
  register_mdrange_policy_function(space, schedule, rank, outer_dir, inner_dir,
                                   index_type_size, index_type_signed, begin,
                                   end, tile);
}

void register_team_policy(const char* space, const char* schedule,
                          std::size_t index_type_size, bool index_type_signed,
                          int team_size, int league_size, int vector_length,
                          const scratch_description& team_scratch,
                          const scratch_description& thread_scratch,
                          int chunk_size) {
  init_internal_functions();
  register_team_policy_function(
      space, schedule, index_type_size, index_type_signed, team_size,
      league_size, vector_length, team_scratch.level0, team_scratch.level1,
      thread_scratch.level0, thread_scratch.level1, chunk_size);
}

bool next_invocation_will_dump(const std::string& kernel_name) {
  init_internal_functions();
  return next_invocation_will_dump_function(kernel_name.c_str());
}

void clear_registered_views() {
  init_internal_functions();
  clear_registered_views_function();
}

void register_view(void* data, const char* space) {
  init_internal_functions();
  register_view_function(data, space);
}

void register_replay_source(const char* source) {
  init_internal_functions();
  register_replay_source_function(source);
}

}  // namespace impl

void add_metadata(const std::string& key, const std::string& value) {
  impl::init_internal_functions();
  impl::add_metadata_function(key.c_str(), value.c_str(), value.size());
}

}  // namespace krepe
