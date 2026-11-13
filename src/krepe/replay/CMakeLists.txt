# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

function(krepe_add_kernel_replayer TARGET_NAME)
  add_library("${TARGET_NAME}")
  target_sources(
    "${TARGET_NAME}"
    PRIVATE kernel_replayer.cpp
    PUBLIC FILE_SET
           replay_headers
           TYPE
           HEADERS
           BASE_DIRS
           "${PROJECT_SOURCE_DIR}/src"
           FILES
           ../replayer.hpp
           allocation.hpp
           cuda_allocation.hpp
           hip_allocation.hpp
           host_allocation.hpp
           kernel_replayer.hpp
           memory_space_type.hpp
           ../common/extended_lambda_utils.hpp
    PRIVATE FILE_SET
            common_headers
            TYPE
            HEADERS
            BASE_DIRS
            "${PROJECT_SOURCE_DIR}/src"
            FILES
            ../common/hdf5_utils.hpp
  )
  target_link_libraries(
    "${TARGET_NAME}" PRIVATE Kokkos::kokkos hdf5::hdf5 hdf5::hdf5_hl $<BUILD_LOCAL_INTERFACE:KREPE::warnings>
  )
endfunction()

krepe_add_kernel_replayer(krepe_replayer)
add_library(KREPE::replayer ALIAS krepe_replayer)
set_target_properties(krepe_replayer PROPERTIES EXPORT_NAME replayer)

if(KREPE_ENABLE_ASAN OR KREPE_ENABLE_UBSAN)
  krepe_add_kernel_replayer(krepe_replayer_sanitized)
  add_library(KREPE::replayer_sanitized ALIAS krepe_replayer_sanitized)
  target_link_libraries(krepe_replayer_sanitized PRIVATE KREPE::sanitizers)
else()
  add_library(KREPE::replayer_sanitized ALIAS krepe_replayer)
endif()

install(TARGETS krepe_replayer EXPORT KREPETargets FILE_SET replay_headers)
