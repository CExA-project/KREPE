#include <Kokkos_Core.hpp>
#include <cassert>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

#include <hdf5.h>
#include <hdf5_hl.h>

#include "kernel_replayer.hpp"
#include "krepe/common/hdf5_utils.hpp"
#include "allocation.hpp"
#include "memory_space_type.hpp"

void remove_cli_args(int& argc, char* argv[], int pos, int n) {
  if (pos + n < argc) {
    for (int i = pos; i < argc - n; i++) {
      std::swap(argv[i], argv[i + n]);
    }
  }
  argc -= n;
}

std::string_view find_flag_argument(int& argc, char* argv[],
                                    std::string_view flag) {
  std::string_view argument;
  for (int i = 1; i < argc; i++) {
    std::string_view current_arg{argv[i]};
    // --flag=argument
    if (current_arg.size() > flag.size() && current_arg.starts_with(flag) &&
        current_arg[flag.size()] == '=') {
      auto equal = current_arg.find_first_of('=');
      argument   = current_arg.substr(equal + 1);
      remove_cli_args(argc, argv, i, 1);
      break;
    }

    // --flag argument
    if (current_arg == flag) {
      if (i + 1 == argc) {
        break;
      }
      argument = argv[i + 1];
      remove_cli_args(argc, argv, i, 2);
      break;
    }
  }

  return argument;
}

void device_init() {
  int device_supports_virtual_address = 0;
#if defined(KOKKOS_ENABLE_CUDA)
  CHECK_CUDA_CALL(cuInit(0));

  CUdevice cuDevice;
  CHECK_CUDA_CALL(cuDeviceGet(&cuDevice, 0));
  CUcontext ctx;
  CHECK_CUDA_CALL(cuCtxCreate(&ctx, 0, cuDevice));

  CHECK_CUDA_CALL(cuDeviceGetAttribute(
      &device_supports_virtual_address,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, cuDevice));
#elif defined(KOKKOS_ENABLE_HIP)
  int hipDevice;
  CHECK_HIP_CALL(hipGetDevice(&hipDevice));

  CHECK_HIP_CALL(hipDeviceGetAttribute(
      &device_supports_virtual_address,
      hipDeviceAttributeVirtualMemoryManagementSupported, hipDevice));
#else
  device_supports_virtual_address = 1;
#endif
  if (!device_supports_virtual_address) {
    throw std::runtime_error(
        "The selected device does not support virtual memory management");
  }
}

static std::vector<char> functor_data;
#if defined(KERNEL_REPLAYER_USE_NVCC_HDL_WORKAROUND)
static std::vector<char> nvcc_inner_lambda_data;

namespace krepe::impl {
void* copy_extended_lambda_inner_lambda(void* inner_lambda_ptr,
                                        std::size_t inner_lambda_size) {
  assert(inner_lambda_size == nvcc_inner_lambda_data.size());

  void* inner_lambda_save = std::malloc(inner_lambda_size);
  std::memcpy(inner_lambda_save, inner_lambda_ptr, inner_lambda_size);
  std::memcpy(inner_lambda_ptr, nvcc_inner_lambda_data.data(),
              inner_lambda_size);

  return inner_lambda_save;
}

void restore_extended_lambda_inner_lambda(void* inner_lambda_ptr,
                                          void* inner_lambda_save) {
  std::memcpy(inner_lambda_ptr, inner_lambda_save,
              nvcc_inner_lambda_data.size());
  std::free(inner_lambda_save);
}
}  // namespace krepe::impl

#endif

namespace krepe {

namespace impl {

static std::unordered_map<std::string, void*>* host_allocations;
static std::unordered_map<std::string, std::unique_ptr<void, void (*)(void*)>>*
    host_output_allocations;
static std::unordered_set<std::string>* host_output_labels;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
static std::unordered_map<std::string, void*>* device_allocations;
static std::unordered_map<std::string, std::unique_ptr<void, void (*)(void*)>>*
    device_output_allocations;
static std::unordered_set<std::string>* device_output_labels;
#endif

static std::unordered_map<std::string, const std::string> metadata;

void* get_allocation(impl::MemorySpaceType memory_space,
                     const std::string& label) {
  if (memory_space == impl::MemorySpaceType::HOST) {
    if (!host_allocations->contains(label)) {
      return nullptr;
    }
    return (*host_allocations)[label];
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    if (!device_allocations->contains(label)) {
      return nullptr;
    }
    return (*device_allocations)[label];
#endif
  }
}

void* get_out_allocation(impl::MemorySpaceType memory_space,
                         const std::string& label) {
  if (memory_space == impl::MemorySpaceType::HOST) {
    auto it = host_output_allocations->find(label);
    if (it == host_output_allocations->end()) {
      return nullptr;
    }
    return it->second.get();
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    auto it = device_output_allocations->find(label);
    if (it == device_output_allocations->end()) {
      return nullptr;
    }
    return it->second.get();
#endif
  }
}

bool has_out_allocation(impl::MemorySpaceType memory_space,
                        const std::string& label) {
  if (memory_space == impl::MemorySpaceType::HOST) {
    return host_output_labels->contains(label);
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    return device_output_labels->contains(label);
#endif
  }
}

void init_functor(char* buffer, std::size_t size) {
  if (functor_data.size() < size) {
    throw std::runtime_error(
        "The stored functor is smaller than the requested functor, expected at "
        "most " +
        std::to_string(functor_data.size()) + "B, got " + std::to_string(size) +
        "B");
  }
  std::memcpy(buffer, functor_data.data(), size);
}

char* allocate_host_buffer(std::size_t size, MemorySpaceType mem) {
  if (mem == MemorySpaceType::HOST) {
    return new char[size];
  } else {
    void* ptr = nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMallocHost(&ptr, size));
#elif defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IMPL_HIP_SAFE_CALL(hipHostMalloc(&ptr, size));
#endif
    return static_cast<char*>(ptr);
  }
}

void free_host_buffer(char* ptr, MemorySpaceType mem) {
  if (mem == MemorySpaceType::HOST) {
    delete[] ptr;
  } else {
#if defined(KOKKOS_ENABLE_CUDA)
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeHost(ptr));
#elif defined(KOKKOS_ENABLE_HIP)
    KOKKOS_IMPL_HIP_SAFE_CALL(hipHostFree(ptr));
#endif
  }
}

using hdf5_iterate_fun_t = std::function<void(std::string, std::string_view,
                                              char*, char*, std::size_t)>;

std::string get_hdf5_string_attribute(hid_t group, const char* name,
                                      const char* attr_name) {
  krepe::hdf5::ScopedHandle attr(
      CHECK_HDF5_ID(
          H5Aopen_by_name(group, name, attr_name, H5P_DEFAULT, H5P_DEFAULT)),
      H5Aclose);

  H5A_info_t attr_info;
  CHECK_HDF5_CALL(H5Aget_info(attr.get(), &attr_info));
  krepe::hdf5::ScopedHandle type(CHECK_HDF5_ID(H5Aget_type(attr.get())),
                                 H5Tclose);
  std::string attribute(attr_info.data_size, '\0');
  CHECK_HDF5_CALL(H5Aread(attr.get(), type.get(), attribute.data()));

  // remove the null-terminator
  attribute.pop_back();
  type.close_checked();
  attr.close_checked();

  return attribute;
}

int get_hdf5_int_attribute(hid_t group, const char* name,
                           const char* attr_name) {
  krepe::hdf5::ScopedHandle attr(
      CHECK_HDF5_ID(
          H5Aopen_by_name(group, name, attr_name, H5P_DEFAULT, H5P_DEFAULT)),
      H5Aclose);

  H5A_info_t attr_info;
  CHECK_HDF5_CALL(H5Aget_info(attr.get(), &attr_info));
  krepe::hdf5::ScopedHandle type(CHECK_HDF5_ID(H5Aget_type(attr.get())),
                                 H5Tclose);
  assert(attr_info.data_size == sizeof(int));
  int attribute;
  CHECK_HDF5_CALL(H5Aread(attr.get(), type.get(), &attribute));

  type.close_checked();
  attr.close_checked();

  return attribute;
}

std::uint64_t get_hdf5_uint64_attribute(hid_t group, const char* name,
                                        const char* attr_name) {
  krepe::hdf5::ScopedHandle attr(
      CHECK_HDF5_ID(
          H5Aopen_by_name(group, name, attr_name, H5P_DEFAULT, H5P_DEFAULT)),
      H5Aclose);

  H5A_info_t attr_info;
  CHECK_HDF5_CALL(H5Aget_info(attr.get(), &attr_info));
  krepe::hdf5::ScopedHandle type(CHECK_HDF5_ID(H5Aget_type(attr.get())),
                                 H5Tclose);
  assert(attr_info.data_size == sizeof(std::uint64_t));
  std::uint64_t attribute;
  CHECK_HDF5_CALL(H5Aread(attr.get(), type.get(), &attribute));

  type.close_checked();
  attr.close_checked();

  return attribute;
}

template <MemorySpaceType target_memory_space>
herr_t get_hdf5_dataset_alloc_info(hid_t group, const char* name,
                                   const H5L_info_t*, void* allocation_set) {
  std::string space = get_hdf5_string_attribute(group, name, "space");
  char* address     = reinterpret_cast<char*>(std::stoull(
      get_hdf5_string_attribute(group, name, "p_data"), nullptr, 16));

  MemorySpaceType memory_space = memory_space_type_from_string(space);
  if (memory_space != target_memory_space) {
    return 0;
  }

  int rank = 0;
  std::string dataset_name(name);
  dataset_name += "/bytes";
  CHECK_HDF5_CALL(H5LTget_dataset_ndims(group, dataset_name.c_str(), &rank));
  std::vector<hsize_t> dims(rank);
  H5T_class_t datatype;
  std::size_t datatype_size;

  CHECK_HDF5_CALL(H5LTget_dataset_info(group, dataset_name.c_str(), dims.data(),
                                       &datatype, &datatype_size));
  // We only deal with arrays of chars
  assert(datatype == H5T_INTEGER);
  assert(datatype_size == 1);

  std::size_t buffer_size =
      std::reduce(dims.begin(), dims.end(), 1, std::multiplies<>{});
  if (buffer_size == 0) {
    return 0;
  }

  auto allocations = reinterpret_cast<std::set<std::pair<char*, std::size_t>>*>(
      allocation_set);
  std::pair<char*, std::size_t> allocation_info =
      get_allocation_address(address, buffer_size, memory_space);
  allocations->insert(allocation_info);

  return 0;
}

herr_t read_hdf5_metadata(hid_t group, const char* name, const H5A_info_t*,
                          void*) {
  std::string attr = get_hdf5_string_attribute(group, ".", name);
  metadata.insert(
      std::pair<std::string, const std::string>(name, std::move(attr)));
  return 0;
}

herr_t allocate_hdf5_dataset(hid_t group, const char* name, const H5L_info_t*,
                             void* allocate_fun) {
  std::string label = get_hdf5_string_attribute(group, name, "label");
  std::string space = get_hdf5_string_attribute(group, name, "space");
  char* address     = reinterpret_cast<char*>(std::stoull(
      get_hdf5_string_attribute(group, name, "p_data"), nullptr, 16));

  int rank = 0;
  std::string dataset_name(name);
  dataset_name += "/bytes";
  CHECK_HDF5_CALL(H5LTget_dataset_ndims(group, dataset_name.c_str(), &rank));
  std::vector<hsize_t> dims(rank);
  H5T_class_t datatype;
  std::size_t datatype_size;

  CHECK_HDF5_CALL(H5LTget_dataset_info(group, dataset_name.c_str(), dims.data(),
                                       &datatype, &datatype_size));
  // We only deal with arrays of chars
  assert(datatype == H5T_INTEGER);
  assert(datatype_size == 1);

  std::size_t buffer_size =
      std::reduce(dims.begin(), dims.end(), 1, std::multiplies<>{});
  if (buffer_size == 0) {
    (*reinterpret_cast<hdf5_iterate_fun_t*>(allocate_fun))(label, space,
                                                           nullptr, nullptr, 0);
    return 0;
  }

  MemorySpaceType memory_space = memory_space_type_from_string(space);
  auto free_buffer             = [memory_space](char* ptr) {
    free_host_buffer(ptr, memory_space);
  };
  std::unique_ptr<char, decltype(free_buffer)> buffer(
      allocate_host_buffer(buffer_size, memory_space), free_buffer);
  CHECK_HDF5_CALL(H5LTread_dataset(group, dataset_name.c_str(),
                                   H5T_NATIVE_UCHAR, buffer.get()));

  (*reinterpret_cast<hdf5_iterate_fun_t*>(allocate_fun))(
      label, space, address, buffer.get(), buffer_size);

  return 0;
}

herr_t allocate_hdf5_output_dataset(hid_t group, const char* name,
                                    const H5L_info_t* info,
                                    void* allocate_fun) {
  if (get_hdf5_int_attribute(group, name, "bytes_dumped") == 0) {
    return 0;
  }

  return allocate_hdf5_dataset(group, name, info, allocate_fun);
}

void read_functor_from_hdf5(hid_t file) {
  int rank = 0;
  CHECK_HDF5_CALL(H5LTget_dataset_ndims(file, "functor/functor", &rank));
  assert(rank == 1);

  hsize_t dim = 0;
  H5T_class_t datatype;
  std::size_t datatype_size;
  CHECK_HDF5_CALL(H5LTget_dataset_info(file, "functor/functor", &dim, &datatype,
                                       &datatype_size));
  // We only deal with arrays of chars
  assert(datatype == H5T_INTEGER);
  assert(datatype_size == 1);

  functor_data.resize(dim);
  CHECK_HDF5_CALL(H5LTread_dataset(file, "functor/functor", H5T_NATIVE_UCHAR,
                                   functor_data.data()));

#if defined(KERNEL_REPLAYER_USE_NVCC_HDL_WORKAROUND)
  CHECK_HDF5_CALL(
      H5LTget_dataset_ndims(file, "functor/nvcc_inner_lambda", &rank));
  assert(rank == 1);

  CHECK_HDF5_CALL(H5LTget_dataset_info(file, "functor/nvcc_inner_lambda", &dim,
                                       &datatype, &datatype_size));
  assert(datatype == H5T_INTEGER);
  assert(datatype_size == 1);

  nvcc_inner_lambda_data.resize(dim);
  CHECK_HDF5_CALL(H5LTread_dataset(file, "functor/nvcc_inner_lambda",
                                   H5T_NATIVE_UCHAR,
                                   nvcc_inner_lambda_data.data()));
#endif
}

std::vector<std::int64_t> read_hdf5_int64_dataset(hid_t root,
                                                  const char* name) {
  int dataset_rank = 0;
  CHECK_HDF5_CALL(H5LTget_dataset_ndims(root, name, &dataset_rank));
  assert(dataset_rank == 1);
  hsize_t dim = 0;
  H5T_class_t datatype;
  std::size_t datatype_size;
  CHECK_HDF5_CALL(
      H5LTget_dataset_info(root, name, &dim, &datatype, &datatype_size));
  assert(datatype == H5T_INTEGER);
  assert(datatype_size == sizeof(std::int64_t));

  std::vector<std::int64_t> res(dim);
  CHECK_HDF5_CALL(H5LTread_dataset(root, name, H5T_NATIVE_INT64, res.data()));

  return res;
}

index_type_var_t get_policy_index_type_from_props(bool index_type_signed,
                                                  std::size_t index_type_size) {
  if (index_type_signed) {
    switch (index_type_size) {
      case 1: return std::int8_t{};
      case 2: return std::int16_t{};
      case 4: return std::int32_t{};
      case 8: return std::int64_t{};
      default:
        throw std::runtime_error("unexpected index type size " +
                                 std::to_string(index_type_size));
    }
  } else {
    switch (index_type_size) {
      case 1: return std::uint8_t{};
      case 2: return std::uint16_t{};
      case 4: return std::uint32_t{};
      case 8: return std::uint64_t{};
      default:
        throw std::runtime_error("unexpected index type size " +
                                 std::to_string(index_type_size));
    }
  }
}

schedule_var_t get_policy_schedule_from_name(const std::string& schedule) {
  if (schedule == "static") {
    return Kokkos::Static{};
  } else if (schedule == "dynamic") {
    return Kokkos::Dynamic{};
  } else {
    throw std::runtime_error("Unexpected schedule type '" + schedule +
                             "' found in the dump");
  }
}

exec_space_var_t get_policy_exec_space_from_name(const std::string& space) {
#if defined(KOKKOS_ENABLE_SERIAL)
  if (space == "Serial") {
    return ExecSpaceTag<Kokkos::Serial>{};
  }
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  if (space == "OpenMP") {
    return ExecSpaceTag<Kokkos::OpenMP>{};
  }
#endif
#if defined(KOKKOS_ENABLE_THREADS)
  if (space == "Threads") {
    return ExecSpaceTag<Kokkos::Threads>{};
  }
#endif
#if defined(KOKKOS_ENABLE_HPX)
  if (space == "HPX") {
    return ExecSpaceTag<Kokkos::HPX>{};
  }
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  if (space == "Cuda") {
    return ExecSpaceTag<Kokkos::Cuda>{};
  }
#endif
#if defined(KOKKOS_ENABLE_HIP)
  if (space == "HIP") {
    return ExecSpaceTag<Kokkos::HIP>{};
  }
#endif
  throw std::runtime_error("No enabled execution space corresponds to " +
                           space);
}

void read_policy_from_hdf5(hid_t file) {
  std::string type = get_hdf5_string_attribute(file, "policy", "type");

  replay_policy = std::make_unique<policy_var_t>();

  if (type == "none") {
    return;
  }

  if (type == "scalar") {
    *replay_policy =
        ScalarPolicyDesc{get_hdf5_uint64_attribute(file, "policy", "end")};
    return;
  }

  const int index_type_signed =
      get_hdf5_int_attribute(file, "policy", "index_type_signed");
  const int index_type_size =
      get_hdf5_int_attribute(file, "policy", "index_type_size");

  const index_type_var_t index_type =
      get_policy_index_type_from_props(index_type_signed, index_type_size);

  const std::string exec_space_name =
      get_hdf5_string_attribute(file, "policy", "space");
  const exec_space_var_t exec_space =
      get_policy_exec_space_from_name(exec_space_name);

  const std::string schedule_name =
      get_hdf5_string_attribute(file, "policy", "schedule");
  const schedule_var_t schedule = get_policy_schedule_from_name(schedule_name);

  if (type == "range") {
    const std::uint64_t begin =
        get_hdf5_uint64_attribute(file, "policy", "begin");
    const std::uint64_t end = get_hdf5_uint64_attribute(file, "policy", "end");
    const int chunk_size = get_hdf5_int_attribute(file, "policy", "chunk_size");
    *replay_policy       = RangePolicyDesc{begin,      end,      chunk_size,
                                     index_type, schedule, exec_space};
  } else if (type == "mdrange") {
    mdrange_rank_var_t policy_rank;
    const int rank = get_hdf5_int_attribute(file, "policy", "rank");
    switch (rank) {
#if KOKKOS_VERSION_GREATER_EQUAL(5, 2, 0)
      case 1: policy_rank = std::integral_constant<int, 1>{}; break;
#endif
      case 2: policy_rank = std::integral_constant<int, 2>{}; break;
      case 3: policy_rank = std::integral_constant<int, 3>{}; break;
      case 4: policy_rank = std::integral_constant<int, 4>{}; break;
      case 5: policy_rank = std::integral_constant<int, 5>{}; break;
      case 6: policy_rank = std::integral_constant<int, 6>{}; break;
      default:
        throw std::runtime_error("Unexpected mdrange policy rank " +
                                 std::to_string(rank));
    }

    Kokkos::Iterate outer_dir;
    const std::string outer_dir_name =
        get_hdf5_string_attribute(file, "policy", "outer_dir");
    if (outer_dir_name == "left") {
      outer_dir = Kokkos::Iterate::Left;
    } else if (outer_dir_name == "right") {
      outer_dir = Kokkos::Iterate::Right;
    } else {
      throw std::runtime_error("Unexpected iteration pattern '" +
                               outer_dir_name + "'");
    }

    Kokkos::Iterate inner_dir;
    const std::string inner_dir_name =
        get_hdf5_string_attribute(file, "policy", "inner_dir");
    if (inner_dir_name == "left") {
      inner_dir = Kokkos::Iterate::Left;
    } else if (inner_dir_name == "right") {
      inner_dir = Kokkos::Iterate::Right;
    } else {
      throw std::runtime_error("Unexpected iteration pattern '" +
                               inner_dir_name + "'");
    }

    std::vector<std::int64_t> begin =
        read_hdf5_int64_dataset(file, "policy/begin");
    std::vector<std::int64_t> end = read_hdf5_int64_dataset(file, "policy/end");
    std::vector<std::int64_t> tile =
        read_hdf5_int64_dataset(file, "policy/tile");

    *replay_policy = MDRangePolicyDesc{begin,       end,       tile,
                                       index_type,  schedule,  exec_space,
                                       policy_rank, outer_dir, inner_dir};
  } else if (type == "team") {
    const int team_size = get_hdf5_int_attribute(file, "policy", "team_size");
    const int league_size =
        get_hdf5_int_attribute(file, "policy", "league_size");
    const int vector_length =
        get_hdf5_int_attribute(file, "policy", "vector_length");
    const int team_scratch_0 =
        get_hdf5_int_attribute(file, "policy", "team_scratch_0");
    const int team_scratch_1 =
        get_hdf5_int_attribute(file, "policy", "team_scratch_1");
    const int thread_scratch_0 =
        get_hdf5_int_attribute(file, "policy", "thread_scratch_0");
    const int thread_scratch_1 =
        get_hdf5_int_attribute(file, "policy", "thread_scratch_1");
    const int chunk_size = get_hdf5_int_attribute(file, "policy", "chunk_size");
    *replay_policy       = TeamPolicyDesc{
        team_size,      league_size,      vector_length,    team_scratch_0,
        team_scratch_1, thread_scratch_0, thread_scratch_1, chunk_size,
        index_type,     schedule,         exec_space};
  } else {
    throw std::runtime_error("Unknown policy type '" + type + "'");
  }
}

std::vector<std::pair<char*, std::size_t>> compute_allocations(
    const std::set<std::pair<char*, std::size_t>>& allocations) {
  if (allocations.empty()) {
    return {};
  }

  std::vector<std::pair<char*, std::size_t>> allocs;
  auto it                              = allocations.begin();
  auto [current_address, current_size] = *(it++);
  for (; it != allocations.end(); ++it) {
    auto [address, size] = *it;
    if (current_address + current_size > address) {
      std::size_t new_size = (address + size) - current_address;
      current_size         = new_size > current_size ? new_size : current_size;
    } else {
      allocs.emplace_back(current_address, current_size);
      current_address = address;
      current_size    = size;
    }
  }

  allocs.emplace_back(current_address, current_size);

  return allocs;
}
}  // namespace impl

ScopeGuard::ScopeGuard(int& argc, char* argv[], bool enable_input_reset)
    : enable_input_reset_(enable_input_reset) {
  std::string_view hdf5_filename =
      find_flag_argument(argc, argv, "--kernel-replayer-dump");
  if (hdf5_filename.data() == nullptr) {
    throw std::runtime_error(
        "The kernel replayer expects the flag --kernel-replayer-dump");
  }

  krepe::hdf5::ScopedHandle file(
      CHECK_HDF5_ID(H5Fopen(hdf5_filename.data(), H5F_ACC_RDONLY, H5P_DEFAULT)),
      H5Fclose);

  std::set<std::pair<char*, std::size_t>> host_allocation_locs;

  hsize_t idx = 0;
  CHECK_HDF5_CALL(H5Literate_by_name(
      file.get(), "in/views", H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
      impl::get_hdf5_dataset_alloc_info<impl::MemorySpaceType::HOST>,
      &host_allocation_locs, H5P_DEFAULT));

  for (auto [address, size] : impl::compute_allocations(host_allocation_locs)) {
    allocate(impl::MemorySpaceType::HOST, address, size);
  }

  // initializing the device driver apis might allocate heap memory, that's why
  // we do it after doing the host allocations
  device_init();

  std::set<std::pair<char*, std::size_t>> device_allocation_locs;

  idx = 0;
  CHECK_HDF5_CALL(H5Literate_by_name(
      file.get(), "in/views", H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
      impl::get_hdf5_dataset_alloc_info<impl::MemorySpaceType::DEVICE>,
      &device_allocation_locs, H5P_DEFAULT));

  for (auto [address, size] :
       impl::compute_allocations(device_allocation_locs)) {
    allocate(impl::MemorySpaceType::DEVICE, address, size);
  }

  impl::hdf5_iterate_fun_t copy_data_wrapper =
      [this](std::string label, std::string_view memory_space, char* address,
             char* data, std::size_t size) {
        impl::MemorySpaceType space =
            impl::memory_space_type_from_string(memory_space);
        if (size == 0) {
          if (space == impl::MemorySpaceType::HOST) {
            host_allocations_[label] = nullptr;
          } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
            device_allocations_[label] = nullptr;
#endif
          }
          return;
        }

        impl::copy_data(space, address, data, size);
        if (enable_input_reset_) {
          input_snapshots_.push_back(
              {space, address, std::vector<char>(data, data + size)});
        }
        if (space == impl::MemorySpaceType::HOST) {
          host_allocations_[label] = address;
        } else {
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
          device_allocations_[label] = address;
#endif
        }
      };

  idx = 0;
  CHECK_HDF5_CALL(H5Literate_by_name(
      file.get(), "in/views", H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
      impl::allocate_hdf5_dataset, &copy_data_wrapper, H5P_DEFAULT));

  idx = 0;
  CHECK_HDF5_CALL(
      H5Aiterate_by_name(file.get(), "metadata", H5_INDEX_NAME, H5_ITER_NATIVE,
                         &idx, impl::read_hdf5_metadata, nullptr, H5P_DEFAULT));

  impl::read_functor_from_hdf5(file.get());

  impl::read_policy_from_hdf5(file.get());

  impl::hdf5_iterate_fun_t allocate_wrapper =
      [this](std::string label, std::string_view memory_space, char*,
             char* data, std::size_t size) {
        allocate_output(label, memory_space, data, size);
      };

  htri_t has_output = H5Lexists(file.get(), "out", H5P_DEFAULT);
  CHECK_HDF5_CALL(has_output);
  if (has_output > 0) {
    has_output = H5Lexists(file.get(), "out/views", H5P_DEFAULT);
    CHECK_HDF5_CALL(has_output);
  }

  if (has_output > 0) {
    idx = 0;
    CHECK_HDF5_CALL(H5Literate_by_name(
        file.get(), "out/views", H5_INDEX_NAME, H5_ITER_NATIVE, &idx,
        impl::allocate_hdf5_output_dataset, &allocate_wrapper, H5P_DEFAULT));
  }

  file.close_checked();

  impl::host_allocations        = &host_allocations_;
  impl::host_output_allocations = &host_output_allocations_;
  impl::host_output_labels      = &host_output_labels_;
#if defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
  impl::device_allocations        = &device_allocations_;
  impl::device_output_allocations = &device_output_allocations_;
  impl::device_output_labels      = &device_output_labels_;
#endif
}

void ScopeGuard::reset_inputs() {
  if (!enable_input_reset_) {
    throw std::runtime_error(
        "Input reset must be enabled when constructing krepe::ScopeGuard");
  }
  if (!Kokkos::is_initialized() || Kokkos::is_finalized()) {
    throw std::runtime_error(
        "reset_inputs can be called only inside Kokkos execution environment");
  }

  Kokkos::fence("KREPE before resetting inputs");
  for (auto& input : input_snapshots_) {
    impl::copy_data(input.memory_space, input.address, input.data.data(),
                    input.data.size());
  }
  Kokkos::fence("KREPE after resetting inputs");
}

std::optional<std::string> get_metadata(const std::string& key) {
  try {
    return impl::metadata.at(key);
  } catch (const std::out_of_range&) {
    return std::nullopt;
  }
}

void ScopeGuard::allocate(impl::MemorySpaceType memory_space, char* address,
                          std::size_t size) {
  if (memory_space == impl::MemorySpaceType::HOST) {
    host_raw_allocations_.emplace_back(memory_space, address, size);
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    device_raw_allocations_.emplace_back(memory_space, address, size);
#endif
  }
}

void ScopeGuard::allocate_output(std::string label,
                                 std::string_view memory_space, char* data,
                                 std::size_t size) {
  if (impl::memory_space_type_from_string(memory_space) ==
      impl::MemorySpaceType::HOST) {
    host_output_labels_.insert(label);
    if (size == 0) {
      return;
    }
    host_output_allocations_.insert_or_assign(
        label, impl::regular_host_allocate(size, data));
  } else {
#if !defined(KERNEL_REPLAYER_HAS_DEVICE_SPACE)
    throw std::runtime_error(
        "Trying to access device allocations but no device space is enabled");
#else
    device_output_labels_.insert(label);
    if (size == 0) {
      return;
    }
    device_output_allocations_.insert_or_assign(
        label, impl::regular_device_allocate(size, data));
#endif
  }
}

}  // namespace krepe
