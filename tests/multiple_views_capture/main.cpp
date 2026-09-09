#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>
#include <krepe/common/hdf5_utils.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace {

using krepe::hdf5::ScopedHandle;

std::string read_string_attribute(hid_t group, const char* name,
                                  const char* attribute) {
  ScopedHandle attr(CHECK_HDF5_ID(H5Aopen_by_name(group, name, attribute,
                                                  H5P_DEFAULT, H5P_DEFAULT)),
                    H5Aclose);
  ScopedHandle type(CHECK_HDF5_ID(H5Aget_type(attr.get())), H5Tclose);
  std::string value(H5Tget_size(type.get()), '\0');
  CHECK_HDF5_CALL(H5Aread(attr.get(), type.get(), value.data()));
  if (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

struct SnapshotEntry {
  std::string name;
  std::string address;
};

std::vector<SnapshotEntry> matching_allocations(hid_t group) {
  H5G_info_t info;
  CHECK_HDF5_CALL(H5Gget_info(group, &info));
  std::vector<SnapshotEntry> entries;
  // Use the same index and iteration order as the replayer.
  for (hsize_t i = 0; i < info.nlinks; ++i) {
    const auto length = H5Lget_name_by_idx(
        group, ".", H5_INDEX_NAME, H5_ITER_NATIVE, i, nullptr, 0, H5P_DEFAULT);
    if (length < 0) throw std::runtime_error("Cannot read snapshot entry");
    std::string name(static_cast<std::size_t>(length) + 1, '\0');
    if (H5Lget_name_by_idx(group, ".", H5_INDEX_NAME, H5_ITER_NATIVE, i,
                           name.data(), name.size(), H5P_DEFAULT) < 0) {
      throw std::runtime_error("Cannot read snapshot entry");
    }
    name.resize(static_cast<std::size_t>(length));
    if (read_string_attribute(group, name.c_str(), "label") == "values") {
      entries.push_back(
          {name, read_string_attribute(group, name.c_str(), "ptr")});
    }
  }
  if (entries.size() != 2 || entries[0].address == entries[1].address) {
    throw std::runtime_error("Expected two distinct allocations named values");
  }
  return entries;
}

void reorder_reference_snapshots() {
  // The test launcher supplies an isolated directory and removes old dumps.
  std::filesystem::path dump;
  for (const auto& entry : std::filesystem::directory_iterator(".")) {
    const auto name = entry.path().filename().string();
    if (name.starts_with("krepe_test_kernel_") && name.ends_with(".h5")) {
      if (!dump.empty()) throw std::runtime_error("Expected one kernel dump");
      dump = entry.path();
    }
  }
  if (dump.empty()) throw std::runtime_error("Kernel dump not found");
  ScopedHandle file(
      CHECK_HDF5_ID(H5Fopen(dump.c_str(), H5F_ACC_RDWR, H5P_DEFAULT)),
      H5Fclose);
  ScopedHandle inputs(
      CHECK_HDF5_ID(H5Gopen2(file.get(), "in/views", H5P_DEFAULT)), H5Gclose);
  ScopedHandle outputs(
      CHECK_HDF5_ID(H5Gopen2(file.get(), "out/views", H5P_DEFAULT)), H5Gclose);
  const auto input = matching_allocations(inputs.get());
  auto output      = matching_allocations(outputs.get());
  if (output[0].address == input[0].address) {
    // Move whole groups: their identity, metadata and reference bytes stay
    // together, while their positions relative to the inputs are exchanged.
    CHECK_HDF5_CALL(H5Lmove(outputs.get(), output[0].name.c_str(),
                            outputs.get(), "swap_reference", H5P_DEFAULT,
                            H5P_DEFAULT));
    CHECK_HDF5_CALL(H5Lmove(outputs.get(), output[1].name.c_str(),
                            outputs.get(), output[0].name.c_str(), H5P_DEFAULT,
                            H5P_DEFAULT));
    CHECK_HDF5_CALL(H5Lmove(outputs.get(), "swap_reference", outputs.get(),
                            output[1].name.c_str(), H5P_DEFAULT, H5P_DEFAULT));
  }
  output = matching_allocations(outputs.get());
  if (output[0].address != input[1].address ||
      output[1].address != input[0].address) {
    throw std::runtime_error(
        "Reference order must be reversed relative to inputs");
  }
  outputs.close_checked();
  inputs.close_checked();
  file.close_checked();
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> A("A", N);
  Kokkos::View<int*> B("values", N);
  Kokkos::View<int*> C("values", N);
  Kokkos::parallel_for(
      "init", N, KOKKOS_LAMBDA(int i) {
        A(i) = i;
        B(i) = i % 32;
      });

  krepe::parallel_for(
      "test_kernel", N, KOKKOS_LAMBDA(int i) {
        B(i) *= 3;
        C(i) = A(i) + B(i);
      });
  Kokkos::fence();

  auto h_C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
  Kokkos::printf("C(5) = %d\n", h_C(5));

  reorder_reference_snapshots();

  return 0;
}
