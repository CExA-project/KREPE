# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(captureless_lambda_capture main.cpp)
target_link_libraries(
  captureless_lambda_capture PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor
)

add_executable(captureless_lambda_capture_replay replay_main.cpp)
target_link_libraries(
  captureless_lambda_capture_replay PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::replayer_sanitized
)

add_replay_test(captureless_lambda_capture COMPARE_OUTPUT)
