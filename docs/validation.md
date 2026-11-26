# Result validation

Ensuring that the replayed kernel produces the right results can be done by
comparing the values of the input allocations (modified by the replayed kernel)
with the output values captured from the orignal program.

## Compare the results

`krepe::compare_views<TArgs...>(label, view_args, compare)` constructs
unmanaged Views over the reference output and the replayed data for a specific
allocation selected by the `label` parameter, it then passes them to the user
provided `compare` function, which should accept two parameters: the reference
view and the view containing the replayed data.

The template arguments to the function are used as template arguments to the
constructed Views, `view_args` is a tuple containing the arguments that should
be passed to the Views' constructor (ususally the dimensions).

The following example compares the `"values"` allocation to its reference, the `expected` and `actual` arguments will be views of type `Kokkos::View<int*, Kokkos::MemoryTraits<Kokkos::Unmanaged>>` with 1024 elements.

```cpp
#include <Kokkos_StdAlgorithms.hpp>
#include <tuple>

// ...

krepe::compare_views<int*>(
    "values", std::make_tuple(1024), [](auto expected, auto actual) {
      if (!Kokkos::Experimental::equal(Kokkos::DefaultExecutionSpace(),
                                       expected, actual)) {
        Kokkos::abort("Replay verification failed");
      }
    });
```

If several allocations share the same label, use
[krepe::get_allocations](reference/replay/get_allocations.md) and pass each
descriptor to [krepe::compare_views](reference/replay/compare_views.md).

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
