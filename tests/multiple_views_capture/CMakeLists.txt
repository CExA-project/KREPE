# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(multiple_views_capture main.cpp)
target_link_libraries(multiple_views_capture PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(multiple_views_capture_replay replay_main.cpp)
target_link_libraries(multiple_views_capture_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(multiple_views_capture)
