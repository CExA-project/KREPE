# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(view_comparison main.cpp)
target_link_libraries(view_comparison PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(view_comparison_replay replay_main.cpp)
target_link_libraries(view_comparison_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(view_comparison FALSE)
