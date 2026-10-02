# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(struct_functor main.cpp)
target_link_libraries(struct_functor PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(struct_functor_replay replay_main.cpp)
target_link_libraries(struct_functor_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(struct_functor FALSE)
