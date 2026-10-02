# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(metadata main.cpp)
target_link_libraries(metadata PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(metadata_replay replay_main.cpp)
target_link_libraries(metadata_replay PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_replayer_sanitized)

add_replay_test(metadata TRUE)
