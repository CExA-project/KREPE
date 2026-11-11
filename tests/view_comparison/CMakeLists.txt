# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(view_comparison main.cpp)
target_link_libraries(view_comparison PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(view_comparison_replay replay_main.cpp)
target_link_libraries(view_comparison_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(view_comparison)
