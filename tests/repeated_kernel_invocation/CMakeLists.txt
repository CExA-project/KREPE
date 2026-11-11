# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(repeated_kernel_invocation main.cpp)
target_link_libraries(
  repeated_kernel_invocation PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor
)

add_executable(repeated_kernel_invocation_replay replay_main.cpp)
target_link_libraries(repeated_kernel_invocation_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(
  repeated_kernel_invocation TOOL_FLAGS --krepe-kernel-invocation=2 REPLAY_FLAGS --krepe-enable-input-reset
)
