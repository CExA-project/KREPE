# HDF5 file layout

Each HDF5 dump stores a kernel's execution context for replay,
In filenames `krepe_<label>_<kernel_id>.h5`.


<div class="hdf5-tree" markdown="1">
<div class="hdf5-root"><code>/</code></div>

<div class="hdf5-children" markdown="1">

<details markdown="1">
<summary><span>Root attributes</span> <span>Kernel identity</span></summary>

<ul class="hdf5-fields">
  <li><code>kernel_label</code> <span>Kokkos kernel label</span></li>
  <li><code>kernel_id</code> <span>Tool-local ID counting all kernel launches, across labels</span></li>
  <li><code>kernel_invocation</code> <span>Invocation number for this label, starting at 1</span></li>
</ul>

</details>

<details markdown="1">
<summary><code>in/</code> <span>Before the kernel</span></summary>

<ul class="hdf5-fields">
  <li><code>active_allocations</code> <span>Number of allocations in this snapshot</span></li>
  <li><code>active_bytes</code> <span>Sum of bounded user-data sizes, including skipped allocations, in bytes</span></li>
</ul>

<div class="hdf5-children" markdown="1">

<details markdown="1">
<summary><code>views/</code> <span>Captured allocations</span></summary>

<div class="hdf5-children" markdown="1">

<details markdown="1">
<summary><code>view_&lt;index&gt;_&lt;label&gt;/</code> <span>One group per allocation</span></summary>

<ul class="hdf5-fields">
  <li><code>label</code> <span>Kokkos allocation label</span></li>
  <li><code>space</code> <span>Memory space reported by the profiling hook</span></li>
  <li><code>ptr</code> <span>Allocation pointer, stored as text</span></li>
  <li><code>p_data</code> <span>User-data pointer after the allocation header, stored as text</span></li>
  <li><code>size</code> <span>Bounded user-data size in bytes; 0 if empty or unbounded</span></li>
  <li><code>reported_size</code> <span>Size reported by the allocation hook, in bytes</span></li>
  <li><code>bytes_dumped</code> <span>1 if bytes is present, including an empty dataset; 0 if skipped</span></li>
  <li><code>skip_reason</code> <span>Present only when bytes_dumped = 0; explains why data was skipped</span></li>
  <li><code>bytes</code> <span>Unsigned-byte dataset; present only when bytes_dumped = 1</span></li>
</ul>

</details>

</div>

</details>

</div>

</details>

<details markdown="1">
<summary><code>out/</code> <span>After the kernel</span></summary>

<ul class="hdf5-fields">
  <li><code>active_allocations</code> <span>Number of allocations in this snapshot</span></li>
  <li><code>active_bytes</code> <span>Sum of bounded user-data sizes, including skipped allocations, in bytes</span></li>
</ul>

<div class="hdf5-children" markdown="1">

<details markdown="1">
<summary><code>views/</code> <span>Captured allocations</span></summary>

<div class="hdf5-children" markdown="1">

<details markdown="1">
<summary><code>view_&lt;index&gt;_&lt;label&gt;/</code> <span>One group per allocation</span></summary>

<ul class="hdf5-fields">
  <li><code>label</code> <span>Kokkos allocation label</span></li>
  <li><code>space</code> <span>Memory space reported by the profiling hook</span></li>
  <li><code>ptr</code> <span>Allocation pointer, stored as text</span></li>
  <li><code>p_data</code> <span>User-data pointer after the allocation header, stored as text</span></li>
  <li><code>size</code> <span>Bounded user-data size in bytes; 0 if empty or unbounded</span></li>
  <li><code>reported_size</code> <span>Size reported by the allocation hook, in bytes</span></li>
  <li><code>bytes_dumped</code> <span>1 if bytes is present, including an empty dataset; 0 if skipped</span></li>
  <li><code>skip_reason</code> <span>Present only when bytes_dumped = 0; explains why data was skipped</span></li>
  <li><code>bytes</code> <span>Unsigned-byte dataset; present only when bytes_dumped = 1</span></li>
</ul>

</details>

</div>

</details>

</div>

</details>

<details markdown="1">
<summary><code>policy/</code> <span>Execution policy</span></summary>

Select the recorded policy type to see its fields

=== "Range"

    - `type` <span>"range"</span>
    - `space` <span>Execution space name</span>
    - `schedule` <span>Scheduling policy: static or dynamic</span>
    - `index_type_size` <span>Index type size, in bytes</span>
    - `index_type_signed` <span>Whether the index type is signed</span>
    - `begin` <span>Start index</span>
    - `end` <span>End index</span>
    - `chunk_size` <span>Chunk size parameter</span>

=== "MDRange"

    - `type` <span>"mdrange"</span>
    - `space` <span>Execution space name</span>
    - `schedule` <span>Scheduling policy: static or dynamic</span>
    - `index_type_size` <span>Index type size, in bytes</span>
    - `index_type_signed` <span>Whether the index type is signed</span>
    - `rank` <span>Number of dimensions</span>
    - `outer_dir` <span>Outer iteration direction: left or right</span>
    - `inner_dir` <span>Inner iteration direction: left or right</span>
    - `begin` <span>int64_t dataset of start indices</span>
    - `end` <span>int64_t dataset of end indices</span>
    - `tile` <span>int64_t dataset of tile dimensions</span>

=== "Team"

    - `type` <span>"team"</span>
    - `space` <span>Execution space name</span>
    - `schedule` <span>Scheduling policy: static or dynamic</span>
    - `index_type_size` <span>Index type size, in bytes</span>
    - `index_type_signed` <span>Whether the index type is signed</span>
    - `team_size` <span>Threads per team, or -1 for AUTO</span>
    - `league_size` <span>Number of teams</span>
    - `vector_length` <span>Vector lanes per team thread, or -1 for AUTO</span>
    - `team_scratch_0` <span>Level 0 scratch bytes per team; -1 if unavailable</span>
    - `team_scratch_1` <span>Level 1 scratch bytes per team; -1 if unavailable</span>
    - `thread_scratch_0` <span>Level 0 scratch bytes per thread; -1 if unavailable</span>
    - `thread_scratch_1` <span>Level 1 scratch bytes per thread; -1 if unavailable</span>
    - `chunk_size` <span>Chunk size parameter</span>

=== "Scalar"

    - `type` <span>"scalar"</span>
    - `end` <span>End index</span>

=== "None"

    - `type` <span>"none" — no policy was recorded</span>

</details>

<details markdown="1">
<summary><code>metadata/</code> <span>User metadata</span></summary>

<ul class="hdf5-fields">
  <li><code>&lt;key&gt;</code> <span>One string attribute per user key/value pair</span></li>
</ul>

</details>

<details markdown="1">
<summary><code>functor/</code> <span>Callable data for replay</span></summary>

<ul class="hdf5-fields">
  <li><code>functor</code> <span>Unsigned-byte dataset of the captured functor or lambda object</span></li>
  <li><code>nvcc_inner_lambda</code> <span>Unsigned-byte dataset; always present, empty unless an NVCC extended host-device lambda was captured</span></li>
</ul>

</details>

</div>

</div>
