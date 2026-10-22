# Kernel replay

Once the [kernel dump](capture.md) has been generated, the kernel can be
replayed in a separate program.

## Prepare your application

The replay program has to be linked with `krepe::kernel_replayer`:

```cmake
find_package(krepe REQUIRED)
target_link_libraries(replay_prog PRIVATE krepe::kernel_replayer Kokkos::kokkos)
```

See [Installation](installation.md) for the KREPE build instructions.

## Prepare the replay program

Start from the [capture example](capture.md) and keep the `scale` kernel.
The dump supplies its input data. Adapt the program as follows:

1. Replace `<krepe/extractor.hpp>` with `<krepe/replayer.hpp>`.
2. Add `krepe::ScopeGuard` before `Kokkos::ScopeGuard` to load the dump.
3. Declare `Kokkos::View<int*> values;` without allocating it, and remove the
   `init` kernel. Keep the same captured variables and their types.

The execution policy is restored from the dump, so we can pass `0` as the
second argument of `krepe::parallel_for`:

```cpp
#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;

  krepe::parallel_for(
      "scale", 0, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
  Kokkos::fence();

  return 0;
}
```

KREPE restores the View captured by the lambda before running `scale`.
The local `values` variable stays empty.

## Run the replay

Pass the dump using the `--kernel-replayer-dump` command-line flag:

```sh
./replay_prog --kernel-replayer-dump=krepe_scale_2.h5
```

Replace `krepe_scale_2.h5` with the path to the HDF5 file generated during
capture.

## Modifying the execution policy

By default, `krepe::parallel_for` uses the execution policy saved in the dump.
The policy argument (`0` in the example above) is ignored. Override it by
passing `krepe::force_policy(your_policy)`:

```cpp
krepe::parallel_for(
    "scale", krepe::force_policy(Kokkos::RangePolicy<>(0, 1024)),
    KOKKOS_LAMBDA(int i) { values(i) *= 2; });
Kokkos::fence();
```

## Parallel_for wrapper

For a kernel captured through another library's wrapper, use
`krepe::replay_functor` to restore the functor's data:

```cpp
my_funky_parallel_for("kernel", 1024,
                      krepe::replay_functor(
                          KOKKOS_LAMBDA(int i) { values(i) *= 2; }));
```

As in [Kernel capture](capture.md#parallel_for-wrapper),
`my_funky_parallel_for` represents your library's wrapper. The execution
policy is not saved in this case, so it must be supplied to the wrapper.

## Accessing the allocations

The local `values` View stays empty during replay. Use the captured allocation's
label with `krepe::get_allocation` to access the active buffer. It initially
contains the captured inputs, then the replayed kernel modifies it.
`krepe::get_out_allocation` returns the reference output recorded during capture.

After the replay and its fence, wrap these pointers in unmanaged Views:

```cpp
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
using AllocationView =
    Kokkos::View<int*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

auto* data = static_cast<int*>(krepe::get_allocation<memory_space>("values"));
auto* reference_data =
    static_cast<int*>(krepe::get_out_allocation<memory_space>("values"));

AllocationView replayed_values(data, 1024);
AllocationView reference_values(reference_data, 1024);
```

This snippet assumes that both the input allocation and its reference output
are present. Check the pointers for `nullptr` before constructing the Views
when either may be missing. KREPE owns these buffers: do not free the pointers
or keep these Views beyond the lifetime of `replay_scope`.

For result checking, see the optional `krepe::compare_views` example in the
[Quickstart](quickstart.md#replay-the-kernel).
