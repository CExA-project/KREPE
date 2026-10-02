# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(range_policy main.cpp)
target_link_libraries(range_policy PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(range_policy_replay replay_main.cpp)
target_link_libraries(range_policy_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(range_policy FALSE)
