# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(duplicate_labels_capture main.cpp)
target_link_libraries(
  duplicate_labels_capture PRIVATE KREPE::warnings KREPE::sanitizers Kokkos::kokkos KREPE::extractor
)

add_executable(duplicate_labels_capture_replay replay_main.cpp)
target_link_libraries(duplicate_labels_capture_replay PRIVATE KREPE::warnings Kokkos::kokkos KREPE::replayer)

add_replay_test(duplicate_labels_capture FALSE)
