# HDF5 Dump Format

The `--krepe-dump-kernel-label=<label>` argument selects one Kokkos kernel label
to dump. For example, `--krepe-dump-kernel-label=sum_values` matches a kernel
launched with the label `"sum_values"`.

For each matching kernel invocation, the tool writes one HDF5 file:

```text
krepe_<kernel-label>_<kernel-id>.h5
```

The `/in` group is written before the kernel runs, and the `/out` group is
appended after it completes. Each group contains one snapshot of all active
user-visible Kokkos allocations.

The root stores these attributes:

```text
kernel_label       # Kokkos label of the matched kernel
kernel_id          # tool-local id assigned to this kernel invocation
kernel_invocation  # allow dumping a specific kernel invocation
```

The `/in` and `/out` groups each store these snapshot specific attributes:

```text
active_allocations # tracked user allocations alive at dump time
active_bytes       # total bounded user-data bytes for those active allocations
```

The `/in/views` and `/out/views` groups contain one subgroup per active
allocation. Each allocation group stores:

```text
label         # Kokkos allocation label
space         # Kokkos memory space name reported by the profiling hook
ptr           # allocation pointer value, stored as text
p_data        # user data pointer after the Kokkos allocation header
size          # bounded user-data bytes, 0 for empty or unbounded allocations
reported_size # size reported by the Kokkos profiling allocation hook
bytes_dumped  # 1 when the bytes dataset was written, 0 when it was skipped
skip_reason   # present only when bytes_dumped is 0
bytes         # byte dataset, present when bytes_dumped is 1
```

The `/metadata` group contains user specified metadata

The `/policy` group contains the type of policy that was saved as well as its
compile-time and runtime attributes:

```
type              # the type of policy stored, or "none" if no policy was recorded
space             # the execution space's name (for every type except scalar)
schedule          # the scheduling policy
index_type_size   # size in bytes of the policy's index_type (for every type except scalar)
index_type_signed # whether the policy's index_type is signed (for every type except scalar)

# type == "scalar"
end # the end index

# type == "range"
begin      # the starting index
end        # the end index
chunk_size # the chunk size parameter

# type == "mdrange"
rank      # the policy's rank
outer_dir # the policy's outer iteration direction
inner_dir # the policy's inner iteration direction
begin     # int64_t dataset storing the starting indices
end       # int64_t dataset storing the end indices
tile      # int64_t dataset storing the tile dimensions

# type == "team"
team_size       # number of threads per team
league_size     # number of teams
vector_length   # number of vector lanes per team thread, or -1 if AUTO
team_scratch_0   # amount of level 0 scratch memory per team, in bytes
team_scratch_1   # amount of level 1 scratch memory per team, in bytes
thread_scratch_0 # amount of level 0 scratch memory per thread, in bytes
thread_scratch_1 # amount of level 1 scratch memory per thread, in bytes
chunk_size      # the chunk size parameter
```
