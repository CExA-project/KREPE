# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(const_constexpr_capture main.cpp)
target_link_libraries(const_constexpr_capture PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(const_constexpr_capture_replay replay_main.cpp)
target_link_libraries(
  const_constexpr_capture_replay PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::replayer_sanitized
)

add_replay_test(const_constexpr_capture COMPARE_OUTPUT)
