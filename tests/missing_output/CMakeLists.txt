# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(missing_output main.cpp)
target_include_directories(missing_output PRIVATE "${PROJECT_SOURCE_DIR}/src/krepe/capture")
target_link_libraries(missing_output PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos krepe_kokkos_hooks)

add_executable(missing_output_replay replay_main.cpp)
target_link_libraries(missing_output_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(missing_output FALSE)
