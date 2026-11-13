# Result validation

Validate a replay by comparing its results with the reference output recorded
during capture. The examples below use the `values` allocation from [Kernel replay](replay.md).

Run these checks after the replayed kernel and its `Kokkos::fence()`, while
`replay_scope` and `kokkos_scope` are still alive.

## Compare the results

`krepe::compare_views` constructs unmanaged Views over the reference output and
the replayed data, then passes them to your comparison function as `expected`
and `actual`, respectively. You supply the allocation label, value type and
dimensions, which must match the captured allocation.

After the replay and its fence, compare all integers:

```cpp
#include <Kokkos_StdAlgorithms.hpp>
#include <tuple>
...
krepe::compare_views<int*>(
    "values", std::make_tuple(1024), [](auto expected, auto actual) {
      if (!Kokkos::Experimental::equal(Kokkos::DefaultExecutionSpace(),
                                       expected, actual)) {
        Kokkos::abort("Replay verification failed");
      }
    });
```

If several allocations share a label, use [krepe::get_allocations](api.md#krepeget_allocations)
and pass each descriptor to [krepe::compare_views](api.md#krepecompare_views).

## Access the reference output

For direct access to the buffers, use `krepe::get_allocation` for the active data
modified by the replay and `krepe::get_out_allocation` for the reference output
recorded during capture.

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
