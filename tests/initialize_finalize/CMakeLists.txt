# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(initialize_finalize main.cpp)
target_link_libraries(initialize_finalize PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(initialize_finalize_replay replay_main.cpp)
target_link_libraries(initialize_finalize_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(initialize_finalize)
