# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(duplicate_labels_capture main.cpp)
target_link_libraries(
  duplicate_labels_capture PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor hdf5::hdf5
)

add_executable(duplicate_labels_capture_replay replay_main.cpp)
target_link_libraries(duplicate_labels_capture_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(duplicate_labels_capture FALSE)
