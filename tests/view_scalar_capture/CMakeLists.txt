# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(view_scalar_capture main.cpp)
target_link_libraries(view_scalar_capture PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(view_scalar_capture_replay replay_main.cpp)
target_link_libraries(view_scalar_capture_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(view_scalar_capture FALSE)
