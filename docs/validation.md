# Result validation

Validate a replay by comparing its results with the reference output recorded
during capture. The examples below use the `values` allocation of 1024 integers
from [Kernel replay](replay.md).

Run these checks after the replayed kernel and its `Kokkos::fence()`, while
`replay_scope` and `kokkos_scope` are still alive.

## Compare the results

`krepe::compare_views` constructs unmanaged Views over the reference output and
the replayed data, then passes them to your comparison function as `expected`
and `actual`, respectively. You supply the allocation label, value type and
dimensions, which must match the captured allocation.

Add these headers to the replay program:

```cpp
#include <Kokkos_StdAlgorithms.hpp>
#include <tuple>
```

After the replay and its fence, compare all 1024 integers:

```cpp
krepe::compare_views<int*>(
    "values", std::make_tuple(1024), [](auto expected, auto actual) {
      if (!Kokkos::Experimental::equal(Kokkos::DefaultExecutionSpace(),
                                       expected, actual)) {
        Kokkos::abort("Replay verification failed");
      }
    });
```

This example uses the default memory and execution spaces, matching the
`Kokkos::View<int*>` in the replay program.

The allocation label must identify a single allocation in the selected memory
space, and the dump must contain its reference output.

## Access the reference output

For direct access to the buffers, use `krepe::get_allocation` for the active data
modified by the replay and `krepe::get_out_allocation` for the reference output
recorded during capture. Wrap the pointers in unmanaged Views:

```cpp
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
using AllocationView =
    Kokkos::View<int*, memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

auto* data = static_cast<int*>(krepe::get_allocation<memory_space>("values"));
auto* reference_data =
    static_cast<int*>(krepe::get_out_allocation<memory_space>("values"));

if (data == nullptr || reference_data == nullptr) {
  Kokkos::abort("Replay data or reference output is missing");
}

AllocationView replayed_values(data, 1024);
AllocationView reference_values(reference_data, 1024);
```

Use `reference_values` as the expected result and `replayed_values` as the actual
result in your comparison. KREPE owns both buffers: do not free the pointers or
keep these Views beyond the lifetime of `replay_scope`.
