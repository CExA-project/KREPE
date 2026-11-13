# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(nested_lambda_capture main.cpp)
target_link_libraries(nested_lambda_capture PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor)

add_executable(nested_lambda_capture_replay replay_main.cpp)
target_link_libraries(nested_lambda_capture_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(nested_lambda_capture)

add_executable(nested_lambda_capture_host host_main.cpp)
target_link_libraries(
  nested_lambda_capture_host PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor
)

add_executable(nested_lambda_capture_host_replay host_replay_main.cpp)
target_link_libraries(
  nested_lambda_capture_host_replay PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::replayer_sanitized
)

add_replay_test(nested_lambda_capture_host)
