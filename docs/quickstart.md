# Quickstart

Capture a kernel that doubles 1024 integers, then replay it in a separate
program.

The [capture program](https://github.com/CExA-project/KREPE/blob/main/examples/kokkos_kernel_example.cpp)
and a [replay program with result checking](https://github.com/CExA-project/KREPE/blob/main/examples/kernel_replay.cpp)
are available in the repository's `examples/` directory.

See [Installation](installation.md) for the prerequisites and build instructions.

## capture.cpp

```cpp
#include <Kokkos_Core.hpp>

#include <krepe/extractor.hpp>

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values("values", N);
  Kokkos::parallel_for(
      "init", N, KOKKOS_LAMBDA(int i) { values(i) = i; });

  // Use KREPE's wrapper to capture this kernel's data and execution policy.
  krepe::parallel_for(
      "scale", N, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```

## replay.cpp

The replay uses the captured data and execution policy. The second argument
of `krepe::parallel_for` is ignored, so we can pass `0`.

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

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.26)
project(krepe_quickstart LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(krepe REQUIRED)

add_executable(capture_prog capture.cpp)
target_link_libraries(capture_prog PRIVATE krepe::kernel_extractor)

add_executable(replay_prog replay.cpp)
target_link_libraries(replay_prog PRIVATE krepe::kernel_replayer Kokkos::kokkos)
```

## Build the example

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/path/to/compiler \
  -DCMAKE_PREFIX_PATH="/path/to/krepe;/path/to/kokkos"
cmake --build build --parallel
```

## Capture the kernel

Load the installed Kokkos tool and select the first invocation of `scale`:

```sh
./build/capture_prog \
  --kokkos-tools-libs=/path/to/krepe/lib/libkrepe.so \
  --kokkos-tools-args="--krepe-dump-kernel-label=scale"
```

The first invocation is captured by default. To capture another one, add
`--krepe-dump-kernel-invocation=N` to `--kokkos-tools-args`.

The tool creates a file named `krepe_scale_<kernel_id>.h5` in the current
directory and reports its path.

## Replay the kernel

Pass the generated dump to the replay program. Replace
`krepe_scale_<kernel_id>.h5` with the dump path reported during capture:

```sh
# Load the captured data and execution policy from the dump.
./build/replay_prog \
  --kernel-replayer-dump="krepe_scale_<kernel_id>.h5"
```

<details markdown="1">
<summary>Verify all 1024 values (optional)</summary>

To check the results, uses `krepe::compare_views` to compare
all 1024 values with the output recorded during capture:

```cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <krepe/replayer.hpp>

#include <iostream>
#include <tuple>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;

  // Replay with the captured inputs and policy (the 0 argument is ignored).
  krepe::parallel_for(
      "scale", 0, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  // Compare the replayed values with the captured output.
  krepe::compare_views<int*>(
      "values", std::make_tuple(1024), [](auto expected, auto actual) {
        if (!Kokkos::Experimental::equal(Kokkos::DefaultExecutionSpace(),
                                         expected, actual)) {
          Kokkos::abort("Replay verification failed");
        }
      });
  std::cout << "Replay verified: 1024 values match.\n";

  return 0;
}
```

</details>
