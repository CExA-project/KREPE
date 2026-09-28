cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(
  KREPE
  VERSION 0.0.1
  DESCRIPTION "Kernel Replay Execution for Performance Evaluation"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KREPE_BUILD_EXAMPLES "Build KREPE examples" ON)
option(KREPE_BUILD_TESTS "Build KREPE tests" OFF)
option(KREPE_ENABLE_WARNINGS "Enable compiler warning options for KREPE targets" ON)
option(KREPE_ENABLE_ASAN "Enable AddressSanitizer instrumentation" OFF)
option(KREPE_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer instrumentation" OFF)
option(KREPE_ENABLE_COMPILER_PLUGINS "Enable the use of compiler plugins if the CXX compiler supports them" ON)
option(KREPE_GCC_PLUGIN_HEADERS_DIR
       "Path to a directory containing the necessary headers to build a gcc plugin (gcc-plugin.h, etc)" ""
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_library(krepe_warnings INTERFACE)

if(KREPE_ENABLE_WARNINGS)
  target_compile_options(
    krepe_warnings INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
  )
endif()

add_library(krepe_sanitizers INTERFACE)

if(KREPE_ENABLE_ASAN OR KREPE_ENABLE_UBSAN)
  target_compile_options(krepe_sanitizers INTERFACE -fno-omit-frame-pointer)
endif()

if(KREPE_ENABLE_ASAN)
  target_compile_options(krepe_sanitizers INTERFACE -fsanitize=address)
  target_link_options(krepe_sanitizers INTERFACE -fsanitize=address)
endif()

if(KREPE_ENABLE_UBSAN)
  target_compile_options(krepe_sanitizers INTERFACE -fsanitize=undefined -fno-sanitize-recover=undefined)
  target_link_options(krepe_sanitizers INTERFACE -fsanitize=undefined)
endif()

set(KREPE_MINIMUM_KOKKOS_VERSION 5.0)

find_package(Kokkos ${KREPE_MINIMUM_KOKKOS_VERSION} REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(src/krepe)

include(CTest)

if(KREPE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(KREPE_BUILD_TESTS)
  add_subdirectory(tests)
endif()

write_basic_package_version_file(krepeConfigVersion.cmake COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/krepeConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/krepeConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/krepe"
)

install(
  EXPORT krepeTargets
  FILE krepeTargets.cmake
  NAMESPACE krepe::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/krepe"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/krepeConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/krepeConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/krepe"
)
