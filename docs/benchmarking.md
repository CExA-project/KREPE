# Benchmarking & auto-tuning

One issue when replaying a `krepe::parallel_for` is that manually measuring the
kernel's execution time by placing timers before and after the `parallel_for`
call will also measure the overhead of the internal operations used to restore
the execution policy and functor. We plan to address this in the future. Note that
Kokkos Tools-based timers will still report the right execution time (that of the
underlying `Kokkos::parallel_for`).

We are also planning to look into integration of KREPE with auto-tuning tools
such as [APEX](https://github.com/UO-OACISS/apex).

## Repeating a replay with the same inputs

It might be useful to be able to replay the kernel multiple times in a single execution of the replay program. If you want to be able to reset input data between each kernel execution, you can do so by enabling input reset from the command line with the `--krepe-enable-input-reset` flag:

```sh
./replay_prog --krepe-dump-file="krepe_scale_<kernel_id>.h5" \
  --krepe-enable-input-reset
```

Replace `krepe_scale_<kernel_id>.h5` with the dump path reported during capture.

You can then call `krepe::reset_inputs()` to restore the input allocations to their initial state.

```cpp
#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  Kokkos::View<int*> values;

  for (int repetition = 0; repetition < 10; ++repetition) {
    krepe::reset_inputs();
    krepe::parallel_for(
        "scale", 0, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
    Kokkos::fence();
  }

  return 0;
}
```
