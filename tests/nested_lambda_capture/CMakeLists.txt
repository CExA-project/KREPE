# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(nested_lambda_capture main.cpp)
target_link_libraries(nested_lambda_capture PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(nested_lambda_capture_replay replay_main.cpp)
target_link_libraries(nested_lambda_capture_replay PRIVATE krepe_warnings Kokkos::kokkos kernel_replayer)

add_replay_test(nested_lambda_capture FALSE)

add_executable(nested_lambda_capture_host host_main.cpp)
target_link_libraries(
  nested_lambda_capture_host PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor
)

add_executable(nested_lambda_capture_host_replay host_replay_main.cpp)
target_link_libraries(
  nested_lambda_capture_host_replay PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_replayer_sanitized
)

add_replay_test(nested_lambda_capture_host FALSE)
