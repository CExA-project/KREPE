# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(repeated_kernel_invocation main.cpp)
target_link_libraries(
  repeated_kernel_invocation PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor
)

add_executable(repeated_kernel_invocation_replay replay_main.cpp)
target_link_libraries(repeated_kernel_invocation_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(repeated_kernel_invocation FALSE "--krepe-dump-kernel-invocation=2")
