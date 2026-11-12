# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(mdrange_policy main.cpp)
target_link_libraries(mdrange_policy PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(mdrange_policy_replay replay_main.cpp)
target_link_libraries(mdrange_policy_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(mdrange_policy)
