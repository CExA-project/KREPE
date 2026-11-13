# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(manual_extraction main.cpp)
target_link_libraries(manual_extraction PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(manual_extraction_replay replay_main.cpp)
target_link_libraries(manual_extraction_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(manual_extraction)
