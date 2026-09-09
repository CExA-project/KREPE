#pragma once

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <variant>
#include <tuple>
#include <Kokkos_Core.hpp>
#include "allocation.hpp"
#include <krepe/common/extended_lambda_utils.hpp>

namespace krepe {

// Non-owning description of an allocation in the active replay. The pointers
// remain valid until its ScopeGuard is destroyed. A null reference can describe
// a valid empty buffer, use has_reference to test availability.
struct ReplayAllocation {
  std::string label;
  std::string memory_space;
  void* data                       = nullptr;
  std::size_t size_bytes           = 0;
  const void* reference_data       = nullptr;
  std::size_t reference_size_bytes = 0;
  bool has_input                   = false;
  bool has_reference               = false;
};

namespace impl {

#if defined(KERNEL_REPLAYER_USE_NVCC_HDL_WORKAROUND)
void* copy_extended_lambda_inner_lambda(void* inner_lambda_ptr,
                                        std::size_t inner_lambda_size);
void restore_extended_lambda_inner_lambda(void* inner_lambda_ptr,
                                          void* inner_lambda_save);
#endif

void init_functor(char* buffer, std::size_t size);
void* get_allocation(impl::MemorySpaceType memory_space,
                     const std::string& label);
void* get_out_allocation(impl::MemorySpaceType memory_space,
                         const std::string& label);
bool has_out_allocation(impl::MemorySpaceType memory_space,
                        const std::string& label);

struct SnapshotAllocation;
struct StoredAllocation {
  char* captured_allocation;
  ReplayAllocation descriptor;
  bool output_seen = false;
  std::unique_ptr<void, void (*)(void*)> reference{nullptr, nullptr};
};

std::vector<ReplayAllocation> get_allocations(
    MemorySpaceType memory_space, std::optional<std::string_view> label);
void validate_comparison(const ReplayAllocation& allocation,
                         MemorySpaceType memory_space);

template <class T>
struct add_unmanaged_trait;

template <unsigned int N>
struct add_unmanaged_trait<Kokkos::MemoryTraits<N>> {
  using type = Kokkos::MemoryTraits<N | Kokkos::Unmanaged>;
};

template <class Policy>
struct ForcePolicy {
  Policy policy;
};

template <class Policy>
constexpr bool is_force_policy_v = false;

template <class Policy>
constexpr bool is_force_policy_v<ForcePolicy<Policy>> = true;

// Variant which is visited at runtime to handle the compile time index type of
// execution policies
using index_type_var_t =
    std::variant<std::int8_t, std::uint8_t, std::int16_t, std::uint16_t,
                 std::int32_t, std::uint32_t, std::int64_t, std::uint64_t>;

// Variant which is visited at runtime to handle the compile time scheduling
// policy of execution policies
using schedule_var_t = std::variant<Kokkos::Static, Kokkos::Dynamic>;

// NOTE: We don't use execution space instances in the variant since we need to
// initialize it before Kokkos is initalized
template <class Space>
struct ExecSpaceTag {
  using space = Space;
};

// Variant which is visited at runtime to handle the execution space of the
// execution policies
using exec_space_var_t = std::variant<std::monostate
#if defined(KOKKOS_ENABLE_SERIAL)
                                      ,
                                      ExecSpaceTag<Kokkos::Serial>
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
                                      ,
                                      ExecSpaceTag<Kokkos::OpenMP>
#endif
#if defined(KOKKOS_ENABLE_THREADS)
                                      ,
                                      ExecSpaceTag<Kokkos::Threads>
#endif
#if defined(KOKKOS_ENABLE_HPX)
                                      ,
                                      ExecSpaceTag<Kokkos::HPX>
#endif
#if defined(KOKKOS_ENABLE_CUDA)
                                      ,
                                      ExecSpaceTag<Kokkos::Cuda>
#endif
#if defined(KOKKOS_ENABLE_HIP)
                                      ,
                                      ExecSpaceTag<Kokkos::HIP>
#endif
                                      >;

// Variant which is visited at runtime to handle the compile time rank of
// MDRangePolicy
using mdrange_rank_var_t = std::variant<
#if KOKKOS_VERSION_GREATER_EQUAL(5, 2, 0)
    std::integral_constant<int, 1>,
#endif
    std::integral_constant<int, 2>, std::integral_constant<int, 3>,
    std::integral_constant<int, 4>, std::integral_constant<int, 5>,
    std::integral_constant<int, 6>>;

struct ScalarPolicyDesc {
  std::size_t N;
};

struct RangePolicyDesc {
  std::uint64_t begin;
  std::uint64_t end;
  int chunk_size;
  index_type_var_t index_type;
  schedule_var_t schedule;
  exec_space_var_t exec_space;
};

struct MDRangePolicyDesc {
  std::vector<std::int64_t> begin;
  std::vector<std::int64_t> end;
  std::vector<std::int64_t> tile;
  index_type_var_t index_type;
  schedule_var_t schedule;
  exec_space_var_t exec_space;
  mdrange_rank_var_t rank;
  Kokkos::Iterate outer_iter_dir;
  Kokkos::Iterate inner_iter_dir;
};

struct TeamPolicyDesc {
  int team_size;
  int league_size;
  int vector_length;
  int team_scratch_0;
  int team_scratch_1;
  int thread_scratch_0;
  int thread_scratch_1;
  int chunk_size;
  index_type_var_t index_type;
  schedule_var_t schedule;
  exec_space_var_t exec_space;
};

using policy_var_t =
    std::variant<std::monostate, ScalarPolicyDesc, RangePolicyDesc,
                 MDRangePolicyDesc, TeamPolicyDesc>;
inline std::unique_ptr<policy_var_t> replay_policy;

template <class ExecSpace, class Schedule, class IndexType>
auto get_range_policy(const IndexType& start, const IndexType& end,
                      int chunk_size) {
  return Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Schedule>,
                             Kokkos::IndexType<IndexType>>(
      start, end, Kokkos::ChunkSize(chunk_size));
}

template <int rank, Kokkos::Iterate outer_dir, Kokkos::Iterate inner_dir,
          class ExecSpace, class Schedule, class IndexType>
auto get_mdrange_policy(const std::vector<std::int64_t>& start,
                        const std::vector<std::int64_t>& end,
                        const std::vector<std::int64_t>& tile) {
  Kokkos::Array<std::int64_t, rank> start_arr;
  Kokkos::Array<std::int64_t, rank> end_arr;
  Kokkos::Array<std::int64_t, rank> tile_arr;
  for (int i = 0; i < rank; i++) {
    start_arr[i] = start[i];
    end_arr[i]   = end[i];
    tile_arr[i]  = tile[i];
  }
  return Kokkos::MDRangePolicy<ExecSpace, Kokkos::Schedule<Schedule>,
                               Kokkos::IndexType<IndexType>,
                               Kokkos::Rank<rank, outer_dir, inner_dir>>(
      start_arr, end_arr, tile_arr);
}

template <class ExecSpace, class Schedule, class IndexType>
auto get_team_policy(int team_size, int league_size, int vector_length,
                     int team_scratch_0, int team_scratch_1,
                     int thread_scratch_0, int thread_scratch_1,
                     int chunk_size) {
  using Policy = Kokkos::TeamPolicy<ExecSpace, Kokkos::Schedule<Schedule>,
                                    Kokkos::IndexType<IndexType>>;
  Policy policy;
  if (team_size < 0) {
    if (vector_length < 0) {
      policy = Policy(league_size, Kokkos::AUTO, Kokkos::AUTO);
    } else {
      policy = Policy(league_size, Kokkos::AUTO, vector_length);
    }
  } else {
    if (vector_length < 0) {
      policy = Policy(league_size, team_size, Kokkos::AUTO);
    } else {
      policy = Policy(league_size, team_size, vector_length);
    }
  }
  // FIXME: querying the scratch size is not supported on all backends for some
  // versions of Kokkos, we pass -1 to indicate that
  if (team_scratch_0 != -1) {
    policy.set_scratch_size(0, Kokkos::PerTeam(team_scratch_0),
                            Kokkos::PerThread(thread_scratch_0));
    policy.set_scratch_size(1, Kokkos::PerTeam(team_scratch_1),
                            Kokkos::PerThread(thread_scratch_1));
  }
  policy.set_chunk_size(chunk_size);
  return policy;
}

template <class Functor, std::size_t N>
struct functor_supports_n_args;

template <class Functor>
struct functor_supports_n_args<Functor, 1> {
  static constexpr bool value = requires(Functor f) { f(0); };
};

template <class Functor>
struct functor_supports_n_args<Functor, 2> {
  static constexpr bool value = requires(Functor f) { f(0, 0); };
};

template <class Functor>
struct functor_supports_n_args<Functor, 3> {
  static constexpr bool value = requires(Functor f) { f(0, 0, 0); };
};

template <class Functor>
struct functor_supports_n_args<Functor, 4> {
  static constexpr bool value = requires(Functor f) { f(0, 0, 0, 0); };
};

template <class Functor>
struct functor_supports_n_args<Functor, 5> {
  static constexpr bool value = requires(Functor f) { f(0, 0, 0, 0, 0); };
};

template <class Functor>
struct functor_supports_n_args<Functor, 6> {
  static constexpr bool value = requires(Functor f) { f(0, 0, 0, 0, 0, 0); };
};

template <class Functor, std::size_t N>
constexpr bool functor_supports_n_args_v =
    functor_supports_n_args<Functor, N>::value;

template <class ExecSpace, class Schedule, class IndexType>
using team_policy_member_t =
    Kokkos::TeamPolicy<ExecSpace, Kokkos::Schedule<Schedule>,
                       Kokkos::IndexType<IndexType>>::member_type;

template <class Functor, class TeamMember>
constexpr bool is_team_functor_v =
    requires(Functor f, TeamMember team) { f(team); };

template <class Functor>
struct ParallelForVisitor {
  const std::string& label;
  Functor functor;

  ParallelForVisitor(const std::string& label, const Functor& functor)
      : label(label), functor(functor) {}

  void operator()(std::monostate) const {
    throw std::runtime_error(
        "Trying to use a replay parallel_for but the replay dump does not "
        "contain an execution policy");
  }

  void operator()(const ScalarPolicyDesc& policy) const {
    if constexpr (functor_supports_n_args_v<Functor, 1>) {
      Kokkos::parallel_for(label, policy.N, functor);
    }
  }

  void operator()(const RangePolicyDesc& policy) const {
    if constexpr (functor_supports_n_args_v<Functor, 1>) {
      std::visit(
          [&]<class IndexType, class Schedule, class ExecSpaceTag>(
              IndexType, Schedule, ExecSpaceTag) {
            if constexpr (!std::is_same_v<ExecSpaceTag, std::monostate>) {
              IndexType begin, end;
              if constexpr (std::is_signed_v<IndexType>) {
                begin = Kokkos::bit_cast<std::int64_t>(policy.begin);
                end   = Kokkos::bit_cast<std::int64_t>(policy.end);
              } else {
                begin = policy.begin;
                end   = policy.end;
              }
              Kokkos::parallel_for(
                  label,
                  impl::get_range_policy<typename ExecSpaceTag::space, Schedule,
                                         IndexType>(begin, end,
                                                    policy.chunk_size),
                  functor);
            }
          },
          policy.index_type, policy.schedule, policy.exec_space);
    }
  }

  void operator()(const MDRangePolicyDesc& policy) const {
    std::visit(
        [&]<int rank, class IndexType, class Schedule, class ExecSpaceTag>(
            std::integral_constant<int, rank>, IndexType, Schedule,
            ExecSpaceTag) {
          if constexpr (functor_supports_n_args_v<Functor, rank> &&
                        !std::is_same_v<ExecSpaceTag, std::monostate>) {
            using enum Kokkos::Iterate;
            // We don't use a variant for the iteration pattern as adding
            // variants to the visit call has a noticeable impact on compile
            // times.
            if (policy.outer_iter_dir == Right) {
              if (policy.inner_iter_dir == Right) {
                auto p = impl::get_mdrange_policy<rank, Right, Right,
                                                  typename ExecSpaceTag::space,
                                                  Schedule, IndexType>(
                    policy.begin, policy.end, policy.tile);
                Kokkos::parallel_for(label, p, functor);
              } else {
                auto p = impl::get_mdrange_policy<rank, Right, Left,
                                                  typename ExecSpaceTag::space,
                                                  Schedule, IndexType>(
                    policy.begin, policy.end, policy.tile);
                Kokkos::parallel_for(label, p, functor);
              }
            } else {  // Left
              if (policy.inner_iter_dir == Right) {
                auto p = impl::get_mdrange_policy<rank, Left, Right,
                                                  typename ExecSpaceTag::space,
                                                  Schedule, IndexType>(
                    policy.begin, policy.end, policy.tile);
                Kokkos::parallel_for(label, p, functor);
              } else {  // Left
                auto p = impl::get_mdrange_policy<rank, Left, Left,
                                                  typename ExecSpaceTag::space,
                                                  Schedule, IndexType>(
                    policy.begin, policy.end, policy.tile);
                Kokkos::parallel_for(label, p, functor);
              }
            }
          }
        },
        policy.rank, policy.index_type, policy.schedule, policy.exec_space);
  }

  void operator()(const TeamPolicyDesc& policy) const {
    std::visit(
        [&]<class IndexType, class Schedule, class ExecSpaceTag>(
            IndexType, Schedule, ExecSpaceTag) {
          if constexpr (!std::is_same_v<ExecSpaceTag, std::monostate>) {
            using policy_member_type =
                team_policy_member_t<typename ExecSpaceTag::space, Schedule,
                                     IndexType>;
            if constexpr (is_team_functor_v<Functor, policy_member_type>) {
              auto p = impl::get_team_policy<typename ExecSpaceTag::space,
                                             Schedule, IndexType>(
                  policy.team_size, policy.league_size, policy.vector_length,
                  policy.team_scratch_0, policy.team_scratch_1,
                  policy.thread_scratch_0, policy.thread_scratch_1,
                  policy.chunk_size);
              Kokkos::parallel_for(label, p, functor);
            }
          }
        },
        policy.index_type, policy.schedule, policy.exec_space);
  }
};
}  // namespace impl

class ScopeGuard {
 private:
  std::vector<impl::StoredAllocation> replay_allocations;
  std::vector<impl::Allocation> host_raw_allocations;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
  std::vector<impl::Allocation> device_raw_allocations;
#endif

  void allocate(impl::MemorySpaceType memory_space, char* address,
                std::size_t size);

  void allocate_output(const impl::SnapshotAllocation& snapshot, char* data);

 public:
  ScopeGuard(int& argc, char* argv[]);
  ScopeGuard(const ScopeGuard&)            = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ~ScopeGuard();
};

std::optional<std::string> get_metadata(const std::string& key);

template <class MemorySpace>
void* get_allocation(const std::string& label) {
  return impl::get_allocation(
      impl::memory_space_type_from_string(MemorySpace::name()), label);
}

template <class MemorySpace>
void* get_out_allocation(const std::string& label) {
  return impl::get_out_allocation(
      impl::memory_space_type_from_string(MemorySpace::name()), label);
}

template <class View, class Functor, class Tuple>
void compare_views(View const& view, Tuple args, Functor&& f) {
  using memory_space = View::memory_space;
  using value_type   = View::value_type;

  value_type* data = static_cast<value_type*>(
      krepe::get_allocation<memory_space>(view.label()));
  if (!impl::has_out_allocation(
          impl::memory_space_type_from_string(memory_space::name()),
          view.label())) {
    throw std::runtime_error("Reference output for view '" + view.label() +
                             "' is not available in the kernel dump");
  }
  value_type* ref_data = static_cast<value_type*>(
      krepe::get_out_allocation<memory_space>(view.label()));

  using ViewType = Kokkos::View<
      typename View::data_type, typename View::array_layout, memory_space,
      typename impl::add_unmanaged_trait<typename View::memory_traits>::type>;

  ViewType actual = std::make_from_tuple<ViewType>(
      std::tuple_cat(std::forward_as_tuple(data), args));
  ViewType expected = std::make_from_tuple<ViewType>(
      std::tuple_cat(std::forward_as_tuple(ref_data), args));

  f(expected, actual);
}

template <class DataType, class... Properties, class Functor, class Tuple>
void compare_views(const std::string& label, Tuple args, Functor&& f) {
  using View         = Kokkos::View<DataType, Properties...>;
  using memory_space = View::memory_space;
  using value_type   = View::value_type;

  value_type* data =
      static_cast<value_type*>(krepe::get_allocation<memory_space>(label));
  if (!impl::has_out_allocation(
          impl::memory_space_type_from_string(memory_space::name()), label)) {
    throw std::runtime_error("Reference output for view '" + label +
                             "' is not available in the kernel dump");
  }
  value_type* ref_data =
      static_cast<value_type*>(krepe::get_out_allocation<memory_space>(label));

  using ViewType = Kokkos::View<
      typename View::data_type, typename View::array_layout, memory_space,
      typename impl::add_unmanaged_trait<typename View::memory_traits>::type>;

  ViewType actual = std::make_from_tuple<ViewType>(
      std::tuple_cat(std::forward_as_tuple(data), args));
  ViewType expected = std::make_from_tuple<ViewType>(
      std::tuple_cat(std::forward_as_tuple(ref_data), args));

  f(expected, actual);
}

// Enumerate all matching input/output records, duplicate labels remain
// separate descriptors. Enumeration order is unspecified.
template <class MemorySpace>
std::vector<ReplayAllocation> get_allocations(const std::string& label) {
  return impl::get_allocations(
      impl::memory_space_type_from_string(MemorySpace::name()), label);
}

template <class MemorySpace>
std::vector<ReplayAllocation> get_allocations() {
  return impl::get_allocations(
      impl::memory_space_type_from_string(MemorySpace::name()), std::nullopt);
}

// Explicit dimensions/layout for structured comparisons. The caller must
// synchronize the replay kernel before inspecting its results.
template <class DataType, class... Properties, class Tuple, class Functor>
decltype(auto) compare_views(const ReplayAllocation& allocation, Tuple args,
                             Functor&& f) {
  using View         = Kokkos::View<DataType, Properties...>;
  using memory_space = typename View::memory_space;
  using value_type   = typename View::non_const_value_type;
  impl::validate_comparison(
      allocation, impl::memory_space_type_from_string(memory_space::name()));

  using ViewType = Kokkos::View<
      typename View::data_type, typename View::array_layout, memory_space,
      typename impl::add_unmanaged_trait<typename View::memory_traits>::type>;
  using ReferenceView = typename ViewType::const_type;
  ViewType actual     = std::make_from_tuple<ViewType>(std::tuple_cat(
      std::make_tuple(static_cast<value_type*>(allocation.data)), args));
  if (actual.span() > allocation.size_bytes / sizeof(value_type)) {
    throw std::runtime_error("Requested view exceeds allocation '" +
                             allocation.label + "'");
  }
  ReferenceView expected = std::make_from_tuple<ReferenceView>(std::tuple_cat(
      std::make_tuple(
          static_cast<const value_type*>(allocation.reference_data)),
      args));
  return std::invoke(std::forward<Functor>(f), expected, actual);
}

// Infer a flat array's extent from the recorded byte count, including padding.
template <class DataType, class... Properties, class Functor>
decltype(auto) compare_views(const ReplayAllocation& allocation, Functor&& f) {
  using View = Kokkos::View<DataType, Properties...>;
  static_assert(View::rank == 1 && View::rank_dynamic == 1,
                "Inferred comparison requires a one-dimensional View with a "
                "dynamic extent");
  using value_type = typename View::non_const_value_type;
  if (allocation.size_bytes % sizeof(value_type) != 0) {
    throw std::runtime_error("Incompatible byte counts for allocation '" +
                             allocation.label + "'");
  }
  return compare_views<DataType, Properties...>(
      allocation, std::make_tuple(allocation.size_bytes / sizeof(value_type)),
      std::forward<Functor>(f));
}

template <class Functor>
Functor replay_functor(const Functor& functor) {
  constexpr int N = sizeof(Functor);

  // In order to revive the to-be-replayed functor, we:
  // - Create a temporary functor and copy construct it from an existing
  //   functor (we have to use the copy constructor since its the only valid
  //   method to construct a lambda)
  // - Save the binary representation of this temporary functor
  // - memcpy the target functor into the temporary functor's memory
  // - Copy construct a new functor f from the patched temporary functor, since
  //   tracking is disabled, if the target functor contained Views, we won't get
  //   a segfault once they are destroyed
  // - memcpy the old data of the temporary functor back and properly destroy it
  // - return f
  Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_disable();

  void* dummy_functor_storage = std::aligned_alloc(alignof(Functor), N);
  Functor* dummy_functor      = new (dummy_functor_storage) Functor(functor);
  void* dummy_functor_buffer_save = std::malloc(N);
  std::memcpy(dummy_functor_buffer_save, dummy_functor_storage, N);

#if defined(KERNEL_REPLAYER_USE_NVCC_HDL_WORKAROUND)
  [[maybe_unused]] void* inner_lambda_ptr  = nullptr;
  [[maybe_unused]] void* inner_lambda_save = nullptr;
  if constexpr (krepe::hdl_utils::lambda_is_hdl<Functor>()) {
    impl::init_functor(static_cast<char*>(dummy_functor_storage),
                       N - sizeof(void*));
    inner_lambda_ptr =
        krepe::hdl_utils::hdl_host_lambda_pointer(*dummy_functor);
    inner_lambda_save = impl::copy_extended_lambda_inner_lambda(
        inner_lambda_ptr,
        krepe::hdl_utils::hdl_host_lambda_size(*dummy_functor));
  } else
#endif
  {
    impl::init_functor(static_cast<char*>(dummy_functor_storage), N);
  }
  Functor f(*dummy_functor);

  std::memcpy(dummy_functor_storage, dummy_functor_buffer_save, N);
  std::free(dummy_functor_buffer_save);
#if defined(KERNEL_REPLAYER_USE_NVCC_HDL_WORKAROUND)
  if constexpr (krepe::hdl_utils::lambda_is_hdl<Functor>()) {
    impl::restore_extended_lambda_inner_lambda(inner_lambda_ptr,
                                               inner_lambda_save);
  }
#endif
  dummy_functor->~Functor();
  std::free(dummy_functor_storage);

  Kokkos::Impl::SharedAllocationRecord<void, void>::tracking_enable();

  return f;
}

/**
 * @brief Wraps an execution policy and allows to override the saved one in a
 * parallel_for.
 */
template <class Policy>
impl::ForcePolicy<Policy> force_policy(Policy&& policy) {
  return impl::ForcePolicy{policy};
}

/**
 * @brief Replays a parallel_for using the execution policy and functors stored
 * in a replay dump.
 *
 * @param label A label forwarded to the underlying Kokkos::parallel_for
 * @param p The policy to use, if it is `force_policy(other_policy)` then
 * `other_policy` will be used, otherwise this parameter will be ignored and the
 * policy stored in the replay dump will be used
 * @param functor A functor with the same signature and the same captures as the
 * functor to replay, its operator() will be used but its data will be replaced
 * by the one stored in the replay dump
 */
template <class Policy, class Functor>
void parallel_for(const std::string& label, [[maybe_unused]] const Policy& p,
                  Functor&& functor) {
  if constexpr (impl::is_force_policy_v<Policy>) {
    Kokkos::parallel_for(label, p.policy, replay_functor(functor));
  } else {
    std::visit(impl::ParallelForVisitor{label, replay_functor(functor)},
               *impl::replay_policy);
  }
}
}  // namespace krepe
