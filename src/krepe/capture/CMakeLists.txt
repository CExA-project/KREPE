# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

add_library(krepe_kokkos_hooks SHARED)
target_sources(
  krepe_kokkos_hooks
  PRIVATE allocation_tracker.cpp memory_copy.cpp view_dump.cpp kokkos_hooks.cpp
  PRIVATE FILE_SET
          common_headers
          TYPE
          HEADERS
          BASE_DIRS
          "${PROJECT_SOURCE_DIR}/src"
          FILES
          ../common/hdf5_utils.hpp
)
target_link_libraries(krepe_kokkos_hooks PRIVATE krepe_warnings krepe_sanitizers)
target_compile_features(krepe_kokkos_hooks PRIVATE cxx_std_20)

get_target_property(KOKKOS_INCLUDE_DIRS Kokkos::kokkos INCLUDE_DIRECTORIES)
if(NOT KOKKOS_INCLUDE_DIRS)
  get_target_property(KOKKOS_INCLUDE_DIRS Kokkos::kokkos INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT KOKKOS_INCLUDE_DIRS)
    message(FATAL_ERROR "Failed to get the kokkos include directory")
  endif()
endif()
target_include_directories(krepe_kokkos_hooks PRIVATE ${KOKKOS_INCLUDE_DIRS})
target_link_libraries(krepe_kokkos_hooks PRIVATE hdf5::hdf5 hdf5::hdf5_hl)

if(KREPE_HAS_COMPILER_PLUGINS)
  target_compile_definitions(krepe_kokkos_hooks PRIVATE KREPE_HAS_COMPILER_PLUGINS)
endif()

set(KREPE_KOKKOS_ALLOCATION_HEADER_SIZE 128)
if(Kokkos_ENABLE_HIP)
  set(KREPE_KOKKOS_ALLOCATION_HEADER_SIZE 256)
endif()
target_compile_definitions(
  krepe_kokkos_hooks PRIVATE KREPE_KOKKOS_ALLOCATION_HEADER_SIZE=${KREPE_KOKKOS_ALLOCATION_HEADER_SIZE}
)

if(Kokkos_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(krepe_kokkos_hooks PRIVATE KREPE_ENABLE_CUDA_DUMP)
  target_link_libraries(krepe_kokkos_hooks PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()
if(Kokkos_ENABLE_HIP)
  target_compile_definitions(krepe_kokkos_hooks PRIVATE KREPE_ENABLE_HIP_DUMP)
  if(TARGET hip::host)
    target_link_libraries(krepe_kokkos_hooks PRIVATE hip::host)
  else()
    target_link_libraries(krepe_kokkos_hooks PRIVATE hip::device)
  endif()
endif()

set_target_properties(krepe_kokkos_hooks PROPERTIES OUTPUT_NAME krepe LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# We don't export the tool target since it's not supposed to be used through CMake
install(TARGETS krepe_kokkos_hooks)
