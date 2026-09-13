add_library(krepe_compiler_plugin INTERFACE)
install(TARGETS krepe_compiler_plugin EXPORT krepeTargets)
set(KREPE_HAS_COMPILER_PLUGINS FALSE PARENT_SCOPE)

if(NOT KREPE_ENABLE_COMPILER_PLUGINS)
  return()
endif()

set(KREPE_USES_NVCC_WRAPPER FALSE)
if(Kokkos_ENABLE_CUDA)
  kokkos_cxx_compiler_cuda_test(KREPE_CXX_COMPILER_BUILDS_CUDA "${CMAKE_CXX_COMPILER}")
  kokkos_compiler_is_nvcc(KREPE_CXX_COMPILER_IS_NVCC "${CMAKE_CXX_COMPILER}")
  if(NOT KREPE_CXX_COMPILER_BUILDS_CUDA OR KREPE_CXX_COMPILER_IS_NVCC)
    set(KREPE_USES_NVCC_WRAPPER TRUE)
  endif()
  unset(KREPE_CXX_COMPILER_BUILDS_CUDA)
endif()

# Since We cannot create a direct depency to the plugin using
# target_link_libraries, we crate a custom command which has the plugin as a
# dependency and creates an empty header that gets included in every target
# which should depend on the plugin. This The header will be regenerated if the
# plugin needs to be rebuilt, which will lead to a recompilation of the
# executables.
macro(add_plugin_dependency TARGET PLUGIN_NAME)
  set(DUMMY_PLUGIN_HEADER "${CMAKE_BINARY_DIR}/dummy.h")
  add_custom_command(OUTPUT "${DUMMY_PLUGIN_HEADER}" COMMAND cmake -E touch "${DUMMY_PLUGIN_HEADER}" DEPENDS ${TARGET})
  add_custom_target(dummy "" DEPENDS "${DUMMY_PLUGIN_HEADER}")
  set(NVCC_EXTRA_FLAG "$<$<BOOL:${KREPE_USES_NVCC_WRAPPER}>:-Xcompiler;>")
  target_compile_options(
    krepe_compiler_plugin
    INTERFACE $<BUILD_INTERFACE:${NVCC_EXTRA_FLAG}-fplugin=$<TARGET_FILE:${TARGET}>;-include;${DUMMY_PLUGIN_HEADER}>
              $<INSTALL_INTERFACE:${NVCC_EXTRA_FLAG}-fplugin=$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/${PLUGIN_NAME}>
  )
  add_dependencies(krepe_compiler_plugin dummy)
endmacro()

#-----------------------------------------------------
# GCC plugin
#-----------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(KREPE_CXX_COMPILER_IS_NVCC)
    # NOTE: This should be modified if we ever need to call the host gcc with multiple flags
    set(GCC_EXE "${CMAKE_CXX_COMPILER};-c;-x;c++;/dev/null;-Xcompiler")
  else()
    set(GCC_EXE "${CMAKE_CXX_COMPILER}")
  endif()

  execute_process(
    COMMAND ${GCC_EXE} -print-file-name=plugin OUTPUT_VARIABLE GCC_PLUGIN_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if("${GCC_PLUGIN_DIR}" STREQUAL plugin OR NOT EXISTS "${GCC_PLUGIN_DIR}")
    message(WARNING "Your GCC installation does not support plugins\nDisabling plugin support")
    return()
  endif()

  if(EXISTS "${GCC_PLUGIN_DIR}/include/gcc-plugin.h")
    set(GCC_PLUGIN_HEADERS_DIR "${GCC_PLUGIN_DIR}/include/")
  elseif(EXISTS "${KREPE_GCC_PLUGIN_HEADERS_DIR}/gcc-plugin.h")
    set(GCC_PLUGIN_HEADERS_DIR "${KREPE_GCC_PLUGIN_HEADERS_DIR}")
  endif()

  if(NOT GCC_PLUGIN_HEADERS_DIR)
    if(NOT KREPE_GCC_PLUGIN_HEADERS_DIR)
      message(
        WARNING
          "Your GCC installation supports plugins, but the plugin headers cannot be found. You can try installing the necessary package or building the headers using the script in scripts/install_gcc_plugin_headers.sh and providing the path to the headers using -DKREPE_GCC_PLUGIN_HEADERS_DIR=<path/to/the/headers>.\nDisabling plugin support"
      )
    else()
      message(
        WARNING
          "The directory provided with -DKREPE_GCC_PLUGIN_HEADERS_DIR (${KREPE_GCC_PLUGIN_HEADERS_DIR}) does not contain gcc-plugin.h.\nDisabling plugin support"
      )
    endif()
    return()
  endif()

  message(STATUS "Enabling the GCC plugin")
  set(KREPE_HAS_COMPILER_PLUGINS TRUE PARENT_SCOPE)

  find_package(GMP REQUIRED)

  add_library(krepe_gcc_plugin MODULE gcc.cpp)
  target_include_directories(krepe_gcc_plugin PRIVATE "${GCC_PLUGIN_HEADERS_DIR}")
  target_link_libraries(krepe_gcc_plugin PRIVATE GMP::GMP)
  target_compile_options(krepe_gcc_plugin PRIVATE "-fno-rtti")
  set_target_properties(krepe_gcc_plugin PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

  add_plugin_dependency(krepe_gcc_plugin libkrepe_gcc_plugin.so)

  install(TARGETS krepe_gcc_plugin DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

#-----------------------------------------------------
# Clang plugin
#-----------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang OR CMAKE_CXX_COMPILER_ID STREQUAL AppleClang OR CMAKE_CXX_COMPILER_ID STREQUAL
                                                                                        CrayClang
)
  # The LLVM package may run try_compile with c files
  enable_language(C)
  find_package(LLVM QUIET)

  if(NOT LLVM_FOUND)
    message(WARNING "Failed to find the LLVM CMake package\nDisabling plugin support")
    return()
  endif()

  include(AddLLVM)
  if(NOT EXISTS "${LLVM_MAIN_INCLUDE_DIR}/clang/AST/AST.h")
    message(
      WARNING
        "Found the LLVM CMake package, but the clang AST headers are not found. Make sure that the necessary packages are installed.\nDisabling plugin support"
    )
    return()
  endif()

  message(STATUS "Enabling the Clang plugin")
  set(KREPE_HAS_COMPILER_PLUGINS TRUE PARENT_SCOPE)

  add_library(krepe_clang_plugin MODULE clang.cpp clang_auto_extract.cpp)
  target_include_directories(krepe_clang_plugin PRIVATE "${LLVM_INCLUDE_DIRS}")
  target_link_directories(krepe_clang_plugin PRIVATE "${LLVM_LIBRARY_DIRS}")
  target_link_libraries(krepe_clang_plugin PRIVATE "${LLVM_LIBRARIES}")
  llvm_update_compile_flags(krepe_clang_plugin)
  string(REGEX REPLACE " " ";" LLVM_DEFS "${LLVM_DEFINITIONS}")
  target_compile_definitions(krepe_clang_plugin PRIVATE "${LLVM_DEFS}")
  set_target_properties(krepe_clang_plugin PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

  add_plugin_dependency(krepe_clang_plugin libkrepe_clang_plugin.so)

  install(TARGETS krepe_clang_plugin DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()
