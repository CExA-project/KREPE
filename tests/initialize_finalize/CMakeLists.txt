# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(initialize_finalize main.cpp)
target_link_libraries(initialize_finalize PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(initialize_finalize_replay replay_main.cpp)
target_link_libraries(initialize_finalize_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(initialize_finalize)
