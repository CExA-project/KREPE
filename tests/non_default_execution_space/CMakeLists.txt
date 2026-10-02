# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(non_default_execution_space main.cpp)
target_link_libraries(
  non_default_execution_space PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor
)

add_executable(non_default_execution_space_replay replay_main.cpp)
target_link_libraries(non_default_execution_space_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(non_default_execution_space FALSE)
