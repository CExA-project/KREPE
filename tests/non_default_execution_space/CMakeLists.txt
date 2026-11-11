# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(non_default_execution_space main.cpp)
target_link_libraries(
  non_default_execution_space PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor
)

add_executable(non_default_execution_space_replay replay_main.cpp)
target_link_libraries(non_default_execution_space_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(non_default_execution_space)
