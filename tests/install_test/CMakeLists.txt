# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

cmake_minimum_required(VERSION 3.26)

project(KREPE-install-test)

find_package(Kokkos REQUIRED)
find_package(KREPE REQUIRED)

add_executable(test main.cpp)
target_link_libraries(test PRIVATE Kokkos::kokkos KREPE::extractor)

add_executable(replay_test replay_main.cpp)
target_link_libraries(replay_test PRIVATE Kokkos::kokkos KREPE::replayer)
