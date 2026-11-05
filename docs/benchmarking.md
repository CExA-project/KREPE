# Benchmarking & auto-tuning

We plan to add a function to measure kernel execution times. Later, we also
plan to simplify integration with [APEX](https://github.com/UO-OACISS/apex).

## Repeating a replay with the same inputs

Enable input reset with the `--kernel-replayer-enable-input-reset` flag:

```sh
./replay_prog --kernel-replayer-dump="krepe_scale_<kernel_id>.h5" \
  --kernel-replayer-enable-input-reset
```

Replace `krepe_scale_<kernel_id>.h5` with the dump path reported during capture.

Call `krepe::reset_inputs()` before each replay to restore the captured inputs.

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
