# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

enable_testing()

set(TEST_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/launcher.sh")

# CTest is unreliable through the Slurm launcher on Ruche. These targets run
# the same replay tests inside a single Slurm allocation.
add_custom_target(krepe_run_replay_tests)

function(add_replay_test TEST_NAME)
  cmake_parse_arguments(PARSE_ARGV 1 arg "COMPARE_OUTPUT" "" "TOOL_FLAGS;REPLAY_FLAGS")
  set(TEST_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/work/${TEST_NAME}")
  file(MAKE_DIRECTORY "${TEST_WORKING_DIRECTORY}")

  if(NOT arg_TOOL_FLAGS)
    set(arg_TOOL_FLAGS "\"\"")
  endif()

  if(NOT arg_REPLAY_FLAGS)
    set(arg_REPLAY_FLAGS "\"\"")
  endif()

  set(TEST_COMMAND
      "${TEST_LAUNCHER}" "$<TARGET_FILE:KREPE::capture>" "$<TARGET_FILE:${TEST_NAME}_exec>"
      "$<TARGET_FILE:${TEST_NAME}_replay>" "${arg_COMPARE_OUTPUT}" "${arg_TOOL_FLAGS}" "${arg_REPLAY_FLAGS}"
  )

  add_test(NAME "${TEST_NAME}" COMMAND ${TEST_COMMAND} WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}")

  # Replay tests with fixed-address host allocations link to the unsanitized
  # replayer, while scalar replay tests use the sanitized replayer variant.
  set_tests_properties("${TEST_NAME}" PROPERTIES LABELS "replay;sanitizer_compatible")

  add_custom_target(
    "krepe_run_${TEST_NAME}"
    COMMAND ${TEST_COMMAND}
    WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}"
    USES_TERMINAL VERBATIM
  )
  add_dependencies(krepe_run_replay_tests "krepe_run_${TEST_NAME}")
endfunction()

function(add_krepe_test NAME)
  cmake_parse_arguments(PARSE_ARGV 1 arg "COMPARE_OUTPUT;ALLOW_SANITIZER" "DIR;PREFIX" "TOOL_FLAGS;REPLAY_FLAGS")

  if(NOT arg_DIR)
    set(arg_DIR "${NAME}")
  endif()

  set(MAIN_FILE "${arg_DIR}/${arg_PREFIX}main.cpp")
  set(REPLAY_MAIN_FILE "${arg_DIR}/${arg_PREFIX}replay_main.cpp")

  add_executable("${NAME}_exec" "${MAIN_FILE}")
  target_link_libraries("${NAME}_exec" PRIVATE KREPE::warnings KREPE::sanitizers KREPE::extractor Kokkos::kokkos)
  add_executable("${NAME}_replay" "${REPLAY_MAIN_FILE}")
  target_link_libraries("${NAME}_replay" PRIVATE KREPE::warnings Kokkos::kokkos)
  if(arg_ALLOW_SANITIZER)
    target_link_libraries("${NAME}_replay" PRIVATE KREPE::sanitizers KREPE::replayer_sanitized)
  else()
    target_link_libraries("${NAME}_replay" PRIVATE KREPE::replayer)
  endif()

  if(arg_COMPARE_OUTPUT)
    set(COMPARE_OUTPUT_ARG COMPARE_OUTPUT)
  endif()
  add_replay_test("${NAME}" ${COMPARE_OUTPUT_ARG} TOOL_FLAGS ${arg_TOOL_FLAGS} REPLAY_FLAGS ${arg_REPLAY_FLAGS})
endfunction()

add_krepe_test(no_capture COMPARE_OUTPUT ALLOW_SANITIZER)
add_krepe_test(scalar_capture COMPARE_OUTPUT ALLOW_SANITIZER)
add_krepe_test(const_constexpr_capture COMPARE_OUTPUT ALLOW_SANITIZER)
add_krepe_test(view_capture)
add_krepe_test(view_comparison)
add_krepe_test(missing_output)
target_include_directories(missing_output_exec PRIVATE "${PROJECT_SOURCE_DIR}/src/krepe/capture")
target_link_libraries(missing_output_exec PRIVATE KREPE::capture)
add_krepe_test(view_scalar_capture)
add_krepe_test(multiple_views_capture)
add_krepe_test(nested_lambda_capture)
add_krepe_test(nested_lambda_capture_host DIR nested_lambda_capture PREFIX host_ ALLOW_SANITIZER)
add_krepe_test(captureless_lambda_capture COMPARE_OUTPUT ALLOW_SANITIZER)
add_krepe_test(captureless_lambda_capture_nested DIR captureless_lambda_capture PREFIX nested_)
add_krepe_test(duplicate_labels_capture)
add_krepe_test(struct_functor)
add_krepe_test(small_struct_functor COMPARE_OUTPUT ALLOW_SANITIZER)
add_krepe_test(overloaded_functor)
add_krepe_test(
  repeated_kernel_invocation TOOL_FLAGS --krepe-kernel-invocation=2 REPLAY_FLAGS --krepe-enable-input-reset
)
add_krepe_test(metadata COMPARE_OUTPUT ALLOW_SANITIZER)
add_krepe_test(manual_extraction)
add_krepe_test(force_policy)
add_krepe_test(range_policy)
add_krepe_test(mdrange_policy)
add_krepe_test(mdrange_iteration_pattern COMPARE_OUTPUT)
add_krepe_test(team_policy)
add_krepe_test(non_default_execution_space)
add_krepe_test(index_type)
add_krepe_test(mdrange_index_type)
add_krepe_test(empty_view ALLOW_SANITIZER)
add_krepe_test(empty_view_without_extent DIR empty_view PREFIX without_extent_ ALLOW_SANITIZER)
add_krepe_test(initialize_finalize)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
  add_krepe_test(gpu_memory_spaces REPLAY_FLAGS --krepe-enable-input-reset)
endif()

add_subdirectory(memory_copy)
