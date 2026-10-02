# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(team_policy main.cpp)
target_link_libraries(team_policy PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(team_policy_replay replay_main.cpp)
target_link_libraries(team_policy_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(team_policy FALSE)
