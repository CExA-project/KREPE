# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_executable(memory_copy_test main.cpp)
target_include_directories(memory_copy_test PRIVATE "${PROJECT_SOURCE_DIR}/src/krepe/capture")
target_link_libraries(memory_copy_test PRIVATE KREPE::warnings KREPE::sanitizers KREPE::capture)

add_test(NAME memory_copy COMMAND memory_copy_test)
set_tests_properties(memory_copy PROPERTIES LABELS "unit;sanitizer_compatible")
