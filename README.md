# KREPE

KREPE (Kernel Replay, Execution for Performance Evaluation) captures the
execution context of Kokkos kernels and replays them in standalone programs for
controlled performance evaluation.

## Build

KREPE requires Kokkos 5.0 or newer.

Clone the repo, then configure and build with CMake

```sh
git clone https://github.com/CExA-project/KREPE.git
```

The library can be included in a CMake project:
```cmake
find_package(krepe REQUIRED)
target_link_libraries(my_program PRIVATE krepe::kernel_extractor)
```
The replayed program should link with `krepe::kernel_replayer`
```cmake
find_package(krepe REQUIRED)
target_link_libraries(my_program PRIVATE krepe::kernel_replayer)
```

## Extraction

In order to extract a kernel from a program, you have to:
1. replace the desired `Kokkos::parallel_for` with `krepe::parallel_for`,
   this will allow to save the execution policy and the kernel's data
2. execute with the `libkrepe.so` kokkos tool

For example, the following program:
```cpp
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  Kokkos::parallel_for("scale", N,
                       KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```
Will become
```cpp
#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  // We replace the Kokkos parallel_for with the one from krepe
  krepe::parallel_for(
      "scale", N, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```

The program has to be linked with `krepe::kernel_extractor`, it then has to be
executed with the following environment variables in order to extract the first
invocation of the kernel named "scale"
```sh
KOKKOS_TOOLS_LIBS=/path/to/libkrepe.so \
KOKKOS_TOOLS_ARGS="--krepe-dump-kernel-label=scale
--krepe-dump-kernel-invocation=1" \
./prog
```

This generates one HDF5 file named `krepe_scale_2.h5`. See the
[HDF5 dump format](docs/hdf5-dumps.md) for the file naming scheme and stored
metadata.

### Parallel_for wrapper

If you are trying to extract a kernel used in a parallel_for wrapper provided
by another library, you can use the `krepe::replay_functor`
function to save the functor's data

```cpp
#include <krepe/extractor.hpp>

int main() {
  Kokkos::View<int*> values("view", N);
  my_funky_parallel_for("kernel", N,
                        krepe::replay_functor(
                            KOKKOS_LAMBDA(int i) { values(i) *= 2; }));
}
```

Note that the execution policy cannot be saved in this case.

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

If you are using GCC and your GCC installation supports plugins but does not
provide the necessary headers to build them, using the
[install_gcc_plugin_headers.sh](./scripts/install_gcc_plugin_headers.sh) script
will generate and install these headers. You can then configure with
`-DKREPE_GCC_PLUGIN_HEADERS_DIR=<path/to/headers/installation>` in order to use
the generated headers.

#### Clang-based compilers

If you are using Clang or a Clang-based compiler, you might need to specify the
path to your llvm installation when configuring using
`-DLLVM_ROOT=<path/to/llvm>`

## Replay

Once the program dump has been generated, the kernel can be replayed in a
separate program. The new program should include the parallel construct call as
well as the functor declaration from the original program and any variable it
depends on. The replayer should also be initialized before Kokkos, using
`krepe::ScopeGuard`.

The program above becomes
```cpp
#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>  // <krepe/extractor.hpp> -> <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  // We initialize the replayer before Kokkos
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  // The execution policy could be ignored, as it will be restored from the dump
  const int N = 1024;
  // We don't care about the values inside the view, we only need it to have the
  // same type as in the original program
  Kokkos::View<int*> values;
  // No need to initialize, the initialized view from the original program is
  // captured in the dump Kokkos::parallel_for(
  //     "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  // we still replace with krepe::parallel_for
  krepe::parallel_for(
      "scale", N, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```

The program has to be linked with `krepe::kernel_replayer`. Pass the dump using
the `--kernel-replayer-dump` command line flag:

```sh
./replay_prog --kernel-replayer-dump=krepe_scale_2.h5
```

### Repeating a replay with the same inputs

```cpp
krepe::ScopeGuard replay_scope(argc, argv, /*enable_input_reset=*/true);
Kokkos::ScopeGuard kokkos_scope(argc, argv);

replay_scope.reset_inputs();
```

This option stores a copy of the inputs in host memory. `reset_inputs()` waits
for previous work and restores those bytes at the same addresses, completing
all copies before returning.

### Modifying the execution policy

By default, `krepe::parallel_for` will use the execution policy
that was saved in the dump. You can override this by passing
`krepe::force_policy(your_policy)` as the execution policy
argument.

### Accessing the allocations

The value of allocations from the original program can be accessed using the
`get_allocation` and `get_out_allocation` for the values before and after the
kernel respectively.

```cpp
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
// Value of `values` before the kernel
int* initial_values_ptr = static_cast<int*>(krepe::get_allocation<memory_space>("values"));
Kokkos::View<int*> initial_values(initial_values_ptr, 1024);
// Value of `values` after the kernel
int* result_values_ptr = static_cast<int*>(krepe::get_out_allocation<memory_space>("values"));
Kokkos::View<int*> result_values(result_values_ptr, 1024);
```

## Limitations

- Only the memory allocations going through Kokkos are captured (e.g. Views, `Kokkos::malloc`)
- Kernels taking a single generic argument are not supported
  (`KOKKOS_LAMBDA(auto i) { ... }`), if you still want to keep your kernel
  generic, you can use concepts to constrain the argument to either an integer or
  a Kokkos team handle:
  - For `RangePolicy`: `KOKKOS_LABMDA(std::integral auto i) { ... }`
  - For `TeamPolicy`: `KOKKOS_LAMBDA(Kokkos::TeamHandle auto team) { ... }`
  Note however that generic kernels are not supported by nvcc anyway
- The `LaunchBounds` and `WorkTag` template arguments for execution policies
  cannot be automatically restored in the replayed program
- Depending on your Kokkos version, the scratch memory parameters for
  `TeamPolicy` may not be automatically restored in the replayed program
