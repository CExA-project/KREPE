# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(mdrange_iteration_pattern main.cpp)
target_link_libraries(mdrange_iteration_pattern PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(mdrange_iteration_pattern_replay replay_main.cpp)
target_link_libraries(mdrange_iteration_pattern_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(mdrange_iteration_pattern TRUE)
