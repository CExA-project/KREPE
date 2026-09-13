#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <Kokkos_Core.hpp>
#include <krepe/common/extended_lambda_utils.hpp>

namespace krepe {
namespace impl {
struct scratch_description {
  int level0;
  int level1;
};

void copy_functor(const unsigned char* functor_data, std::size_t functor_size);
void copy_functor(const unsigned char* functor_data, std::size_t functor_size,
                  const unsigned char* inner_functor_data,
                  std::size_t inner_functor_size);
void register_scalar_policy(std::uint64_t N);
void register_range_policy(const char* space, const char* schedule,
                           std::size_t index_type_size, bool index_type_signed,
                           std::uint64_t begin, std::uint64_t end,
                           int chunk_size);
void register_mdrange_policy(const char* space, const char* schedule,
                             std::size_t rank, const char* outer_dir,
                             const char* inner_dir, std::size_t index_type_size,
                             bool index_type_signed, const std::int64_t* begin,
                             const std::int64_t* end, const std::int64_t* tile);
void register_team_policy(const char* space, const char* schedule,
                          std::size_t index_type_size, bool index_type_signed,
                          int team_size, int league_size, int vector_length,
                          const scratch_description& team_scratch,
                          const scratch_description& thread_scratch,
                          int chunk_size);
bool next_invocation_will_dump(const std::string& kernel_name);
void clear_registered_views();
void register_view(void* data, const char* space);
void register_replay_source(const char* source);

template <std::integral IndexType>
constexpr std::pair<std::size_t, bool> get_index_type_props() {
  return {sizeof(IndexType), std::is_signed_v<IndexType>};
}

template <std::integral IndexType>
constexpr std::uint64_t index_type_to_u64(IndexType N) {
  if constexpr (std::is_signed_v<IndexType>) {
    std::int64_t lN = N;
    return Kokkos::bit_cast<std::uint64_t>(lN);
  } else {
    return N;
  }
}

inline const char* iteration_pattern_to_string(Kokkos::Iterate direction) {
  switch (direction) {
    case Kokkos::Iterate::Left: return "left";
    case Kokkos::Iterate::Right: return "right";
    case Kokkos::Iterate::Default:
      throw std::runtime_error(
          "default iteration pattern should already be resolved at this point");
  }
  // We get a "control reaches end of non-void function" warning
  return "";
}

template <class Schedule>
const char* schedule_to_string() {
  if constexpr (std::is_same_v<Schedule, Kokkos::Schedule<Kokkos::Static>>) {
    return "static";
  } else if constexpr (std::is_same_v<Schedule,
                                      Kokkos::Schedule<Kokkos::Dynamic>>) {
    return "dynamic";
  } else {
    // nvcc doesn't like static_assert(false, ...)
    static_assert(!std::is_same_v<Schedule, Schedule>,
                  "Invalid scheduling policy");
  }
}

inline void register_bounds(const std::size_t upper_bound) {
  register_scalar_policy(static_cast<std::uint64_t>(upper_bound));
}

template <class... Args>
void register_bounds(const Kokkos::RangePolicy<Args...>& policy) {
  using Policy = Kokkos::RangePolicy<Args...>;
  const auto [idx_type_size, idx_type_signed] =
      get_index_type_props<typename Policy::index_type>();
  const std::uint64_t begin = index_type_to_u64(policy.begin());
  const std::uint64_t end   = index_type_to_u64(policy.end());
  register_range_policy(Policy::execution_space::name(),
                        schedule_to_string<typename Policy::schedule_type>(),
                        idx_type_size, idx_type_signed, begin, end,
                        policy.chunk_size());
}

template <class... Args>
void register_bounds(const Kokkos::MDRangePolicy<Args...>& policy) {
  using Policy = Kokkos::MDRangePolicy<Args...>;

  // point_type and tile_type are always std::int64_t
  static_assert(
      std::is_same_v<typename Policy::point_type::value_type, std::int64_t>);
  static_assert(
      std::is_same_v<typename Policy::tile_type::value_type, std::int64_t>);

  const auto [idx_type_size, idx_type_signed] =
      get_index_type_props<typename Policy::index_type>();
  const char* outer_dir = iteration_pattern_to_string(Policy::outer_direction);
  const char* inner_dir = iteration_pattern_to_string(Policy::inner_direction);
  register_mdrange_policy(policy.m_space.name(),
                          schedule_to_string<typename Policy::schedule_type>(),
                          policy.rank, outer_dir, inner_dir, idx_type_size,
                          idx_type_signed, policy.m_lower.data(),
                          policy.m_upper.data(), policy.m_tile.data());
}

template <class... Args>
void register_bounds(const Kokkos::TeamPolicy<Args...>& policy) {
  using Policy = Kokkos::TeamPolicy<Args...>;
  const auto [idx_type_size, idx_type_signed] =
      get_index_type_props<typename Policy::index_type>();

  // Before Kokkos 5.3, {team,thread}_scratch_size are only available for Cuda,
  // HIP and SYCL team policies.
  constexpr bool supports_querying_scratch_size =
#if KOKKOS_VERSION_GREATER_EQUAL(5, 3, 0)
      true
#elif defined(KOKKOS_ENABLE_CUDA)
      std::is_same_v<typename Policy::execution_space, Kokkos::Cuda>
#elif defined(KOKKOS_ENABLE_HIP)
      std::is_same_v<typename Policy::execution_space, Kokkos::HIP>
#elif defined(KOKKOS_ENABLE_SYCL)
      std::is_same_v<typename Policy::execution_space, Kokkos::SYCL>
#else
      false
#endif
      ;

  scratch_description team_scratch{-1, -1};
  scratch_description thread_scratch{-1, -1};

  if constexpr (supports_querying_scratch_size) {
    team_scratch.level0   = policy.team_scratch_size(0);
    team_scratch.level1   = policy.team_scratch_size(1);
    thread_scratch.level0 = policy.thread_scratch_size(0);
    thread_scratch.level1 = policy.thread_scratch_size(1);
  }

  int team_size = policy.impl_auto_team_size() ? -1 : policy.team_size();
  int vector_length =
      policy.impl_auto_vector_length() ? -1 : policy.impl_vector_length();
  register_team_policy(Policy::execution_space::name(),
                       schedule_to_string<typename Policy::schedule_type>(),
                       idx_type_size, idx_type_signed, team_size,
                       policy.league_size(), vector_length, team_scratch,
                       thread_scratch, policy.chunk_size());
}

}  // namespace impl

/**
 * @brief Copies the functor inside the dump so that the replayer can use it
 * later
 */
template <class Functor>
Functor replay_functor(Functor&& functor) {
#if defined(KERNEL_REPLAYER_USE_NVCC_HDL_WORKAROUND)
  if constexpr (krepe::hdl_utils::lambda_is_hdl<Functor>()) {
    impl::copy_functor(reinterpret_cast<const unsigned char*>(&functor),
                       sizeof(functor),
                       reinterpret_cast<const unsigned char*>(
                           krepe::hdl_utils::hdl_host_lambda_pointer(functor)),
                       krepe::hdl_utils::hdl_host_lambda_size(functor));
  } else
#endif
  {
    impl::copy_functor(reinterpret_cast<const unsigned char*>(&functor),
                       sizeof(Functor));
  }
  return functor;
}

/**
 * @brief Calls a parallel_for and saves the execution policy and the functor.
 */
template <class Policy, class Functor>
void parallel_for(const std::string& label, const Policy& policy,
                  Functor&& functor) {
  // We store the result of `next_invocation_will_dump` in a variable as the
  // first tatement of the function so that the plugins can find it easily
  const bool will_dump = impl::next_invocation_will_dump(label);
  if (will_dump) {
    impl::register_bounds(policy);
    Kokkos::parallel_for(label, policy, replay_functor(functor));
  } else {
    Kokkos::parallel_for(label, policy, functor);
  }
}

void add_metadata(const std::string& key, const std::string& value);
}  // namespace krepe
