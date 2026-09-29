# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(mdrange_policy main.cpp)
target_link_libraries(mdrange_policy PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(mdrange_policy_replay replay_main.cpp)
target_link_libraries(mdrange_policy_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(mdrange_policy FALSE)
