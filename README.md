# KREPE

KREPE (Kernel Replay Execution for Performance Evaluation) captures the
execution context of Kokkos kernels and replays them in standalone programs for
controlled performance evaluation.

See the [documentation](https://cexa-project.github.io/KREPE/) and
[Quickstart](https://cexa-project.github.io/KREPE/quickstart/) for complete
guides. Documentation sources are also available in [docs/](docs/).

## Build

KREPE requires Kokkos 5.0 or newer, a C++20 compiler, CMake 3.26 or newer,
and HDF5 development headers and libraries (C and `HL` components).

Clone the repo, then follow the
[installation guide](https://cexa-project.github.io/KREPE/installation/)
to configure and build with CMake:

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
target_link_libraries(my_program PRIVATE krepe::kernel_replayer Kokkos::kokkos)
```

## Extraction

In order to extract a kernel from a program, you have to:

1. replace the desired `Kokkos::parallel_for` with `krepe::parallel_for`,
   this will allow to save the execution policy and the kernel's data
2. execute with the `libkrepe.so` kokkos tool

For example:

```cpp
#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  // Use KREPE's wrapper to capture this kernel's data and execution policy.
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
KOKKOS_TOOLS_ARGS="--krepe-dump-kernel-label=scale --krepe-dump-kernel-invocation=1" \
./prog
```

This generates one HDF5 file named `krepe_scale_<kernel_id>.h5`.
See the [HDF5 file layout](https://cexa-project.github.io/KREPE/hdf5-dumps/)
for the file structure and stored metadata.

### Parallel_for wrapper

If you are trying to extract a kernel used in a parallel_for wrapper provided
by another library, you can use the `krepe::replay_functor`
function to save the functor's data

```cpp
my_funky_parallel_for("kernel", N,
                      krepe::replay_functor(
                          KOKKOS_LAMBDA(int i) { values(i) *= 2; }));
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

See the [compiler plugin setup](https://cexa-project.github.io/KREPE/installation/#compiler-plugins)
for GCC and Clang requirements.

## Replay

Once the program dump has been generated, the kernel can be replayed in a
separate program. The new program should include the parallel construct call as
well as the functor declaration from the original program and any variable it
depends on. The replayer should also be initialized before Kokkos, using
`krepe::ScopeGuard`.

The program above becomes:

```cpp
#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;

  // Replay with the captured inputs and policy (the 0 argument is ignored).
  krepe::parallel_for(
      "scale", 0, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```

The program has to be linked with `krepe::kernel_replayer`. Pass the dump using
the `--kernel-replayer-dump` command line flag:

```sh
./replay_prog --kernel-replayer-dump="krepe_scale_<kernel_id>.h5"
```

Replace `krepe_scale_<kernel_id>.h5` with the filename reported during capture.
Complete [capture](examples/kokkos_kernel_example.cpp) and
[replay](examples/kernel_replay.cpp) examples are available in the repository.

### Repeating a replay with the same inputs

Enable `--kernel-replayer-enable-input-reset`, then call `krepe::reset_inputs()`
before each replay to restore the captured inputs.
See [Benchmarking & auto-tuning](https://cexa-project.github.io/KREPE/benchmarking/)
for the input reset option and a replay loop example.

### Modifying the execution policy

By default, `krepe::parallel_for` will use the execution policy
that was saved in the dump. You can override this by passing
`krepe::force_policy(your_policy)` as the execution policy
argument.

### Accessing the allocations

`krepe::get_allocation` returns the active replay buffer, while
`krepe::get_out_allocation` returns the reference output recorded during capture.
See [Accessing the allocations](https://cexa-project.github.io/KREPE/replay/#accessing-the-allocations)
for an example using unmanaged Views.

## Limitations

- Only the memory allocations going through Kokkos are captured (e.g. Views, `Kokkos::malloc`)
- Kernels taking a single generic argument are not supported
  (`KOKKOS_LAMBDA(auto i) { ... }`), if you still want to keep your kernel
  generic, you can use concepts to constrain the argument to either an integer or
  a Kokkos team handle:
  - For `RangePolicy`: `KOKKOS_LAMBDA(std::integral auto i) { ... }`
  - For `TeamPolicy`: `KOKKOS_LAMBDA(Kokkos::TeamHandle auto team) { ... }`
  Note however that generic kernels are not supported by nvcc anyway
- The `LaunchBounds` and `WorkTag` template arguments for execution policies
  cannot be automatically restored in the replayed program
- Depending on your Kokkos version, the scratch memory parameters for
  `TeamPolicy` may not be automatically restored in the replayed program

## License

[![License](https://img.shields.io/badge/License-MIT-blue)](https://spdx.org/licenses/MIT.html)

The full license statement used in all headers is available [here](LICENSE).
