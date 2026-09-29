// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

#pragma once

#include <hdf5.h>
#include <source_location>
#include <sstream>
#include <stdexcept>

namespace krepe::hdf5 {

inline void check_call(herr_t status, const char* expr, const char* file,
                       int line) {
  if (status >= 0) {
    return;
  }

  std::stringstream os;
  os << file << ":" << line << ": The call " << expr << " failed";
  throw std::runtime_error(os.str());
}

inline hid_t check_id(hid_t id, const char* expr, const char* file, int line) {
  if (id >= 0) {
    return id;
  }

  std::stringstream os;
  os << file << ":" << line << ": The call " << expr << " failed";
  throw std::runtime_error(os.str());
}

class ScopedHandle {
 public:
  ScopedHandle(hid_t id, herr_t (*close)(hid_t)) : id_(id), close_(close) {}
  ScopedHandle(const ScopedHandle&)            = delete;
  ScopedHandle& operator=(const ScopedHandle&) = delete;

  ~ScopedHandle() {
    if (valid()) {
      close_(id_);
    }
  }

  hid_t get() const { return id_; }

  void close_checked() {
    if (!valid()) {
      return;
    }
    const hid_t id = id_;
    check_call(close_(id), "close_(id)", __FILE__, __LINE__);
    id_ = H5I_INVALID_HID;
  }

 private:
  bool valid() const { return id_ >= 0; }

  hid_t id_;
  herr_t (*close_)(hid_t);
};

}  // namespace krepe::hdf5

#define CHECK_HDF5_CALL(expr)                                            \
  ::krepe::hdf5::check_call((expr), #expr,                               \
                            std::source_location::current().file_name(), \
                            std::source_location::current().line())
#define CHECK_HDF5_ID(expr)                                            \
  ::krepe::hdf5::check_id((expr), #expr,                               \
                          std::source_location::current().file_name(), \
                          std::source_location::current().line())
