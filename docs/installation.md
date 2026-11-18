# Installation

Build and install KREPE with CMake, then link it to the application whose
kernels you want to capture or replay.

## Before you start

KREPE requires [Kokkos](https://github.com/kokkos/kokkos) 5.0 or newer.

You also need:

- A compiler supporting C++20.
- CMake 3.26 or newer.
- HDF5 development headers and libraries, including the C and high-level
  (HL) components.

## Build and install KREPE

Clone the repo, then configure and build with CMake

```sh
git clone https://github.com/CExA-project/KREPE.git
cd KREPE

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/compiler \
  -DCMAKE_INSTALL_PREFIX=/path/to/install/prefix \
  -DKokkos_ROOT=/path/to/kokkos/install
cmake --build build --parallel
cmake --install build
```

If CMake cannot find your HDF5 installation, add
`-DHDF5_ROOT=/path/to/hdf5` to the configure command above.

## Build options

KREPE supports the following CMake options:

| Option | Default | Purpose |
| --- | --- | --- |
| `KREPE_BUILD_EXAMPLES` | `OFF` | Build the capture and replay examples. |
| `KREPE_BUILD_TESTS` | `OFF` | Build the tests. |
| `KREPE_ENABLE_WARNINGS` | `ON` | Enable compiler warnings. |
| `KREPE_ENABLE_ASAN` | `OFF` | Enable AddressSanitizer instrumentation in the library and the tests. |
| `KREPE_ENABLE_UBSAN` | `OFF` | Enable UndefinedBehaviorSanitizer instrumentation in the library and the tests. |
| `KREPE_ENABLE_COMPILER_PLUGINS` | `ON` | Enable a compiler plugin (see below) when supported by the compiler. |
| `KREPE_GCC_PLUGIN_HEADERS_DIR` | Not set | Path to GCC plugin headers when they cannot be found automatically. |

### Compiler plugins

By default, if the compiler supports it, we build compiler plugins which allow
us to have fine-grained control over which allocations are dumped (instead of
dumping every allocation up to the target kernel launch). They work by
introspecting the functor to search for Views contained inside of it, allowing
us to only export allocations belonging to views used inside the functor.
Plugins are currently supported for GCC, NVCC using GCC as host compiler, Clang
and Clang-based compilers (e.g. hipcc) except Intel LLVM.

If needed, compiler plugins can be disabled with `-DKREPE_ENABLE_COMPILER_PLUGINS=OFF`.

#### GCC

The GCC plugin requires GMP and plugin headers matching the compiler.

If you are using GCC and your GCC installation supports plugins but does not
provide the necessary headers to build them, using the
[install_gcc_plugin_headers.sh](https://github.com/CExA-project/KREPE/blob/main/scripts/install_gcc_plugin_headers.sh) script
will generate and install these headers. You can then configure with
`-DKREPE_GCC_PLUGIN_HEADERS_DIR=<path/to/headers/installation>` in order to use
the generated headers.

You can check if your GCC installation supports plugins by ensuring that `gcc -print-file-name=plugin` prints a path to an existing directory (and not just `plugin`) containing some `lib...plugin.so` shared libraries.

#### Clang-based compilers

The clang plugin requires LLVM (and Clang) development files for your clang version.

You might need to specify the path to your llvm installation when configuring using
`-DLLVM_ROOT=<path/to/llvm>`
