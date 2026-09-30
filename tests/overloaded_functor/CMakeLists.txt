# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(overloaded_functor main.cpp)
target_link_libraries(overloaded_functor PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(overloaded_functor_replay replay_main.cpp)
target_link_libraries(overloaded_functor_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(overloaded_functor FALSE)
