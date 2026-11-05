# Kernel capture

In order to extract a kernel from a program, you have to:

1. replace the desired `Kokkos::parallel_for` with `krepe::parallel_for`,
   this will allow to save the execution policy and the kernel's data
2. execute with the `libkrepe.so` kokkos tool

## Prepare your application

The library can be included in a CMake project:

```cmake
find_package(krepe REQUIRED)
target_link_libraries(my_program PRIVATE krepe::kernel_extractor)
```

See [Installation](installation.md) for the KREPE build instructions.

## Instrument the kernel

Replace `Kokkos::parallel_for` with `krepe::parallel_for` for the kernel you
want to capture. For example:

```cpp
#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", values.size(), KOKKOS_LAMBDA(int i) { values(i) = i; });

  // Kokkos::parallel_for(
  //     "scale", N, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  krepe::parallel_for(
      "scale", N, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```

## Run the capture

The program has to be linked with `krepe::kernel_extractor`, it then has to be
executed with the following environment variables in order to extract the first
invocation of the kernel named "scale"

```sh
KOKKOS_TOOLS_LIBS=/path/to/libkrepe.so \
KOKKOS_TOOLS_ARGS="--krepe-dump-kernel-label=scale --krepe-dump-kernel-invocation=1" \
./prog
```

Alternatively, use Kokkos command-line arguments:

```sh
./prog \
  --kokkos-tools-libs=/path/to/libkrepe.so \
  --kokkos-tools-args="--krepe-dump-kernel-label=scale --krepe-dump-kernel-invocation=1"
```

This generates one HDF5 file named `krepe_scale_<kernel_id>.h5`.
See the [HDF5 file layout](hdf5-dumps.md) for
the naming rules, file structure and stored metadata.
The file is written to the current working directory.

### Parallel_for wrapper

If you are trying to extract a kernel used in a parallel_for wrapper provided
by another library, you can use the `krepe::replay_functor`
function to save the functor's data

```cpp
my_funky_parallel_for("kernel", N,
                      krepe::replay_functor(
                          KOKKOS_LAMBDA(int i) { values(i) *= 2; }));
```

Here, `my_funky_parallel_for` represents your library's wrapper. Use this call
in the initialized program above and select
`--krepe-dump-kernel-label=kernel` at runtime.

Note that the execution policy cannot be saved in this case.
