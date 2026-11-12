# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(mdrange_iteration_pattern main.cpp)
target_link_libraries(
  mdrange_iteration_pattern PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor
)

add_executable(mdrange_iteration_pattern_replay replay_main.cpp)
target_link_libraries(mdrange_iteration_pattern_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(mdrange_iteration_pattern COMPARE_OUTPUT)
