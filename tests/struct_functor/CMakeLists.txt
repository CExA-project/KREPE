# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(struct_functor main.cpp)
target_link_libraries(struct_functor PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(struct_functor_replay replay_main.cpp)
target_link_libraries(struct_functor_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(struct_functor)
