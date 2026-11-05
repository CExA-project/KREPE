# Installation

Build and install KREPE with CMake, then link it to the application whose
kernels you want to capture or replay.

## Before you start

KREPE requires Kokkos 5.0 or newer.

You also need:

- CMake 3.26 or newer and a compiler supporting C++20.
- HDF5 development headers and libraries, including the C and high-level
  (`HL`) components.

## Build and install KREPE

Clone the repo, then configure and build with CMake

```sh
git clone https://github.com/CExA-project/KREPE.git
cd KREPE

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/compiler \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build --parallel
cmake --install build
```

If CMake cannot find your HDF5 installation, add
`-DHDF5_ROOT=/path/to/hdf5` to the configure command above.

## Build options

To adjust the KREPE build, add `-DOPTION=VALUE` to its configure command:

| Option | Default | Purpose |
| --- | --- | --- |
| `KREPE_BUILD_EXAMPLES` | `ON` | Build the capture and replay examples. |
| `KREPE_BUILD_TESTS` | `OFF` | Build tests. |
| `KREPE_ENABLE_WARNINGS` | `ON` | Enable compiler warnings. |
| `KREPE_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer instrumentation. |
| `KREPE_ENABLE_UBSAN` | `OFF` | Enable UndefinedBehaviorSanitizer instrumentation. |
| `KREPE_ENABLE_COMPILER_PLUGINS` | `ON` | Enable a compiler plugin when supported. |
| `KREPE_GCC_PLUGIN_HEADERS_DIR` | Not set | Path to GCC plugin headers when they cannot be found automatically. |

### Compiler plugins

By default, if the compiler supports it, we build compiler plugins which allow
us to have fine-grained control over which allocations are dumped (instead of
dumping every allocation up to the current kernel launch). They work by
introspecting the functor to search for Views contained inside of it, allowing
to only export allocations which correspond to one of the functor's views.
Plugins are currently supported for GCC, NVCC using GCC as host compiler, Clang
and Clang-based compilers (e.g. hipcc).

If needed, compiler plugins can be disabled with `-DKREPE_ENABLE_COMPILER_PLUGINS=OFF`.

#### GCC

The GCC plugin requires GMP and plugin headers matching the compiler.

If you are using GCC and your GCC installation supports plugins but does not
provide the necessary headers to build them, using the
[install_gcc_plugin_headers.sh](https://github.com/CExA-project/KREPE/blob/main/scripts/install_gcc_plugin_headers.sh) script
will generate and install these headers. You can then configure with
`-DKREPE_GCC_PLUGIN_HEADERS_DIR=<path/to/headers/installation>` in order to use
the generated headers.

#### Clang-based compilers

Install LLVM and Clang development files matching your compiler.

If you are using Clang or a Clang-based compiler, you might need to specify the
path to your llvm installation when configuring using
`-DLLVM_ROOT=<path/to/llvm>`
