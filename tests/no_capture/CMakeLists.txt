# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(no_capture main.cpp)
target_link_libraries(no_capture PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(no_capture_replay replay_main.cpp)
target_link_libraries(
  no_capture_replay PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_replayer_sanitized
)

add_replay_test(no_capture TRUE)
