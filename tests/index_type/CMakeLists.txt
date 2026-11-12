# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(index_type main.cpp)
target_link_libraries(index_type PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(index_type_replay replay_main.cpp)
target_link_libraries(index_type_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(index_type)
