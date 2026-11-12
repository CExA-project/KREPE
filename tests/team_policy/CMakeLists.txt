# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(team_policy main.cpp)
target_link_libraries(team_policy PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(team_policy_replay replay_main.cpp)
target_link_libraries(team_policy_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(team_policy)
