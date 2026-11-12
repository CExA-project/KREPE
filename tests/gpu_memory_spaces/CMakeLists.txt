# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(gpu_memory_spaces main.cpp)
target_link_libraries(gpu_memory_spaces PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(gpu_memory_spaces_replay replay_main.cpp)
target_link_libraries(gpu_memory_spaces_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(gpu_memory_spaces REPLAY_FLAGS --krepe-enable-input-reset)
