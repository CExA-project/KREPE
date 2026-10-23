# Limitations

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
