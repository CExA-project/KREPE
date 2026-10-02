# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(gpu_memory_spaces main.cpp)
target_link_libraries(gpu_memory_spaces PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(gpu_memory_spaces_replay replay_main.cpp)
target_link_libraries(gpu_memory_spaces_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(gpu_memory_spaces FALSE)
