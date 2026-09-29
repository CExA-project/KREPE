# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(manual_extraction main.cpp)
target_link_libraries(manual_extraction PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(manual_extraction_replay replay_main.cpp)
target_link_libraries(manual_extraction_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(manual_extraction "")
