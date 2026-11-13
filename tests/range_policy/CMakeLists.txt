# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(range_policy main.cpp)
target_link_libraries(range_policy PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(range_policy_replay replay_main.cpp)
target_link_libraries(range_policy_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(range_policy)
