# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_library(krepe_extractor)
add_library(KREPE::extractor ALIAS krepe_extractor)
set_target_properties(krepe_extractor PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME extractor)
target_sources(
  krepe_extractor
  PRIVATE kernel_extractor.cpp
  PUBLIC FILE_SET
         extract_headers
         TYPE
         HEADERS
         BASE_DIRS
         "${PROJECT_SOURCE_DIR}/src"
         FILES
         ../extractor.hpp
         kernel_extractor.hpp
         ../common/extended_lambda_utils.hpp
)
target_link_libraries(
  krepe_extractor PRIVATE $<BUILD_LOCAL_INTERFACE:KREPE::warnings> $<BUILD_LOCAL_INTERFACE:KREPE::sanitizers>
)
target_link_libraries(krepe_extractor PUBLIC Kokkos::kokkos)
if(Kokkos_ENABLE_LIBDL)
  target_compile_definitions(krepe_extractor PRIVATE _GNU_SOURCE)
  target_link_libraries(krepe_extractor PRIVATE ${CMAKE_DL_LIBS})
endif()
target_link_libraries(krepe_extractor PUBLIC krepe_compiler_plugin)

install(TARGETS krepe_extractor EXPORT KREPETargets FILE_SET extract_headers)
