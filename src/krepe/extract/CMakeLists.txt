add_library(kernel_extractor)
set_target_properties(kernel_extractor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_sources(
  kernel_extractor
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
  kernel_extractor PRIVATE $<BUILD_LOCAL_INTERFACE:krepe_warnings> $<BUILD_LOCAL_INTERFACE:krepe_sanitizers>
)
target_link_libraries(kernel_extractor PUBLIC Kokkos::kokkos)
if(Kokkos_ENABLE_LIBDL)
  target_compile_definitions(kernel_extractor PRIVATE _GNU_SOURCE)
  target_link_libraries(kernel_extractor PRIVATE ${CMAKE_DL_LIBS})
endif()
target_link_libraries(kernel_extractor PUBLIC krepe_compiler_plugin)

install(TARGETS kernel_extractor EXPORT krepeTargets FILE_SET extract_headers)
