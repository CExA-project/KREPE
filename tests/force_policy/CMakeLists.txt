# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(force_policy main.cpp)
target_link_libraries(force_policy PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(force_policy_replay replay_main.cpp)
target_link_libraries(force_policy_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(force_policy "")
