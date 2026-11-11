# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(overloaded_functor main.cpp)
target_link_libraries(overloaded_functor PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(overloaded_functor_replay replay_main.cpp)
target_link_libraries(overloaded_functor_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(overloaded_functor)
