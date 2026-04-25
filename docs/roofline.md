# Roofline Notes: Reduction

Reduction is usually memory-bandwidth bound for large `n`. Each input element
contributes about one floating-point add, but every element must be read from
global memory. That gives very low arithmetic intensity, so performance is
normally limited by how quickly the GPU can feed data to the reduction rather
than by peak FLOP/s.

## Basic Model

For an input of `n` float32 values:

```text
bytes_read = n * sizeof(float)
work_flops ~= n - 1 additions
arithmetic_intensity ~= work_flops / bytes_read
                     ~= (n - 1) / (4 * n)
                     ~= 0.25 FLOP/byte
```

The benchmark reports effective bandwidth from an estimated global-memory byte
count:

```text
effective_bandwidth_gb_s =
  estimated_global_bytes_moved / latency_seconds / 1e9
```

Estimated bytes differ by variant because the implementations move different
amounts of global memory. For the custom staged kernels, the estimate includes
each pass over global partials:

```text
pass_bytes = (current_elements_read + partial_elements_written) * sizeof(float)
```

The estimate also includes the final device-to-device copy into the output
scalar. This matters most for `naive_global`, which repeatedly writes and rereads
large intermediate arrays. Shared-memory and warp-shuffle variants create far
fewer global partials per pass, so their estimated traffic is lower.

For CUB, the benchmark uses conservative minimum traffic:

```text
cub_minimum_bytes = (n + 1) * sizeof(float)
```

That is the input read plus one output write. CUB's exact internal traffic is
implementation-dependent and opaque from this benchmark, so the repo avoids
pretending to know it.

## What Roofline Should Answer

For each reduction variant, record:

```text
Is the kernel memory bound or compute bound?
What achieved bandwidth did it reach?
How close is it to CUB?
Did the optimization reduce launches, memory traffic, divergence, or stalls?
```

## Expected Reasoning, Without Fake Numbers

- `naive_global` should spend extra time on repeated kernel launches and global
  memory traffic.
- `shared_interleaved` should reduce global traffic but still show inefficient
  control flow from interleaved addressing.
- `shared_sequential` should improve the shared-memory reduction tree by keeping
  active threads contiguous.
- `warp_shuffle` should reduce shared-memory traffic for intra-warp reductions.
- `vectorized_float4` should reduce load instruction overhead for aligned chunks
  while preserving a safe scalar tail.
- `cub_baseline` should be the strongest reference point because CUB uses years
  of tuned reductions, architecture-specific choices, and careful launch policy.

Fill measured bandwidth only after running the benchmark.

# Roofline Notes: Softmax

Softmax is not a pure bandwidth test because each element usually participates
in at least one `expf`. Still, many practical row-wise softmax shapes are
strongly shaped by memory traffic, row reductions, and synchronization overhead.

## Basic Model

For a matrix with `rows * cols` float32 elements:

```text
elements = rows * cols
matrix_bytes = elements * sizeof(float)
```

The v1 benchmark estimates global traffic as:

```text
naive_bytes = 3 * matrix_bytes
stable_two_pass_bytes = 4 * matrix_bytes
block_reduce_bytes = 4 * matrix_bytes
warp_small_row_bytes = 4 * matrix_bytes
```

`naive` reads the input to sum exponentials, reads again to normalize, and writes
the output. Stable variants read once for max, once for sum, once for
normalization, and write output.

The reported effective bandwidth is:

```text
effective_bandwidth_gb_s =
  estimated_bytes_moved / latency_seconds / 1e9
```

This estimate ignores register traffic, shared-memory traffic, and the true
cost of exponentials. It is useful for comparing staged kernels, not for
claiming a hardware roofline limit.

## Expected Behavior

- Small rows can benefit from one warp per row because warp shuffle reductions
  avoid block-wide synchronization.
- Large rows usually need more than one warp of cooperation; block-level
  reductions give more parallelism per row.
- For large enough matrices, launch overhead matters less and memory/reduction
  efficiency should dominate.
- Framework or Triton softmax may still win through fusion, row-width-specific
  policies, and deeper architecture tuning.

## Measured Softmax Evidence

Source: `results/rtx_5060_ti/softmax_results.csv`.

| shape | fastest variant | median us | highest estimated bandwidth | bandwidth GB/s |
| --- | --- | --- | --- | --- |
| `4096 x 128` | `warp_small_row` | `10.36799978` | `warp_small_row` | `809.0864366` |
| `4096 x 512` | `warp_small_row` | `20.25599964` | `warp_small_row` | `1656.518197` |
| `4096 x 1024` | `warp_small_row` | `45.64800113` | `warp_small_row` | `1470.138064` |
| `1024 x 4096` | `warp_small_row` | `48.49600047` | `warp_small_row` | `1383.802032` |

These bandwidth values are benchmark-side estimates using the formula above,
not physical DRAM counters. They count logical matrix traffic implied by the
kernel structure. Nsight DRAM throughput is recorded separately in
`docs/nsight_analysis.md`.

The small-row results match the roofline expectation that reducing row
synchronization and memory rereads matters. The `1024 x 4096` result also favors
`warp_small_row` in this run, but that is a measured outcome for this
implementation rather than a general rule; a more complete roofline study would
repeat runs and profile the same large shape for both `block_reduce` and
`warp_small_row`.
