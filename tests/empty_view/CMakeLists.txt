# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(empty_view main.cpp)
target_link_libraries(empty_view PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(empty_view_replay replay_main.cpp)
target_link_libraries(
  empty_view_replay PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_replayer_sanitized
)

add_replay_test(empty_view FALSE)

add_executable(empty_view_without_extent without_extent_main.cpp)
target_link_libraries(empty_view_without_extent PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_extractor)

add_executable(empty_view_without_extent_replay without_extent_replay_main.cpp)
target_link_libraries(
  empty_view_without_extent_replay PRIVATE krepe_warnings krepe_sanitizers Kokkos::kokkos kernel_replayer_sanitized
)

add_replay_test(empty_view_without_extent FALSE)
