# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(scalar_capture main.cpp)
target_link_libraries(scalar_capture PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(scalar_capture_replay replay_main.cpp)
target_link_libraries(
  scalar_capture_replay PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_replayer_sanitized
)

add_replay_test(scalar_capture TRUE)
