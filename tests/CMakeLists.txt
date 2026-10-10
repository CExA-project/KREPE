# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

enable_testing()

set(TEST_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/launcher.sh")

# CTest is unreliable through the Slurm launcher on Ruche. These targets run
# the same replay tests inside a single Slurm allocation.
add_custom_target(krepe_run_replay_tests)

function(add_replay_test TEST_NAME COMPARE_OUTPUT)
  set(TEST_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/work/${TEST_NAME}")
  file(MAKE_DIRECTORY "${TEST_WORKING_DIRECTORY}")

  if(COMPARE_OUTPUT)
    set(COMPARE_OUTPUT TRUE)
  else()
    set(COMPARE_OUTPUT FALSE)
  endif()

  set(TEST_COMMAND "${TEST_LAUNCHER}" "$<TARGET_FILE:krepe_kokkos_hooks>" "$<TARGET_FILE:${TEST_NAME}>"
                   "$<TARGET_FILE:${TEST_NAME}_replay>" "${COMPARE_OUTPUT}" ${ARGN}
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

add_subdirectory(no_capture)
add_subdirectory(scalar_capture)
add_subdirectory(view_capture)
add_subdirectory(view_comparison)
add_subdirectory(missing_output)
add_subdirectory(view_scalar_capture)
add_subdirectory(multiple_views_capture)
add_subdirectory(nested_lambda_capture)
add_subdirectory(duplicate_labels_capture)
add_subdirectory(struct_functor)
add_subdirectory(small_struct_functor)
add_subdirectory(overloaded_functor)
add_subdirectory(repeated_kernel_invocation)
add_subdirectory(metadata)
add_subdirectory(manual_extraction)
add_subdirectory(force_policy)
add_subdirectory(range_policy)
add_subdirectory(mdrange_policy)
add_subdirectory(mdrange_iteration_pattern)
add_subdirectory(team_policy)
add_subdirectory(non_default_execution_space)
add_subdirectory(index_type)
add_subdirectory(empty_view)

if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
  add_subdirectory(gpu_memory_spaces)
endif()
