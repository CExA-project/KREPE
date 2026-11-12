# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(small_struct_functor main.cpp)
target_link_libraries(small_struct_functor PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(small_struct_functor_replay replay_main.cpp)
target_link_libraries(
  small_struct_functor_replay PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::replayer_sanitized
)

add_replay_test(small_struct_functor COMPARE_OUTPUT)
