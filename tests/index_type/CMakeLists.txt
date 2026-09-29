# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(index_type main.cpp)
target_link_libraries(index_type PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(index_type_replay replay_main.cpp)
target_link_libraries(index_type_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(index_type FALSE)
