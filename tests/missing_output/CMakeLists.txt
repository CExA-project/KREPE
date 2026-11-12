# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(missing_output main.cpp)
target_include_directories(missing_output PRIVATE "${PROJECT_SOURCE_DIR}/src/krepe/capture")
target_link_libraries(missing_output PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::capture)

add_executable(missing_output_replay replay_main.cpp)
target_link_libraries(missing_output_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(missing_output)
