# Benchmarking & auto-tuning

## Repeating a replay with the same inputs

Enable input reset with the `--kernel-replayer-enable-input-reset` flag:

```sh
./replay_prog --kernel-replayer-dump=krepe_scale_2.h5 \
  --kernel-replayer-enable-input-reset
```

Call `krepe::reset_inputs()` before each replay to restore the captured inputs:

```cpp
#include <Kokkos_Core.hpp>

#include <krepe/replayer.hpp>

int main(int argc, char* argv[]) {
  krepe::ScopeGuard replay_scope(argc, argv);
  Kokkos::ScopeGuard kokkos_scope(argc, argv);

  const int N = 1024;
  Kokkos::View<int*> values;

  for (int repetition = 0; repetition < 10; ++repetition) {
    krepe::reset_inputs();
    krepe::parallel_for(
        "scale", N, KOKKOS_LAMBDA(int i) { values(i) *= 2; });
    Kokkos::fence();
  }

  return 0;
}
```

This option keeps a copy of the inputs in host memory.
The reset waits for previous work and restores those bytes at the same
addresses.
