# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(kokkos_kernel_example kokkos_kernel_example.cpp)

target_link_libraries(kokkos_kernel_example PRIVATE krepe_sanitizers Kokkos::kokkos kernel_extractor)
target_compile_features(kokkos_kernel_example PRIVATE cxx_std_20)

add_executable(kernel_replay kernel_replay.cpp)
target_link_libraries(kernel_replay PRIVATE Kokkos::kokkos kernel_replayer)
