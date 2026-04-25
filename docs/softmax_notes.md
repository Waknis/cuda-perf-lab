# Softmax Notes

Softmax converts each row of logits into a probability distribution:

```text
softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
```

The benchmark computes row-wise softmax for a row-major float32 matrix. Each row
is independent, which maps naturally to one thread, one warp, or one block per
row depending on row width.

## Numerical Stability

Optimized variants subtract the row maximum before exponentiation:

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

This prevents overflow when logits are large and is the standard form used in
AI inference kernels. The `naive` variant uses bounded inputs and exists as a
simple baseline, not as production numerics.

## Reduction Structure

Softmax contains two row reductions:

- max reduction to find the row maximum
- sum reduction to normalize exponentials

After those reductions, every element is normalized and written to output.

## Memory Traffic

The v1 bandwidth estimate counts global matrix traffic:

```text
naive estimated bytes ~= 2 input reads + 1 output write
stable/optimized estimated bytes ~= 3 input reads + 1 output write
```

The estimate does not count register/shared-memory traffic and does not model
the cost of `expf`. This keeps the reported bandwidth honest but simple.

## Warp Per Row

`warp_small_row` assigns one warp to each row. This is useful when rows are
small enough that a warp can cover the row with a few lane-strided iterations.
It targets `cols <= 1024`, but the implementation is safe for larger rows by
having lanes stride across the row. Concretely, lane `i` visits columns `i`,
`i + 32`, `i + 64`, and so on during the max, sum, and output passes.

## Block Per Row

`block_reduce` assigns one CUDA block to each row. This is better suited to
moderate or large rows because 256 threads can cooperate on the max and sum
reductions before writing normalized output.

## Why Softmax Is Often Memory Sensitive

Softmax is not a pure memory copy because `expf` has real compute cost. In many
practical inference shapes, though, performance is strongly shaped by memory
traffic, row reductions, and synchronization overhead. The useful question is
not just "how many exponentials?" but also how often the row is reread and how
efficiently reductions are performed.

## RTX 5060 Ti Benchmark Evidence

Source: `results/rtx_5060_ti/softmax_results.csv`.

| shape | fastest variant | median us | best bandwidth variant | bandwidth GB/s |
| --- | --- | --- | --- | --- |
| `4096 x 128` | `warp_small_row` | `10.36799978` | `warp_small_row` | `809.0864366` |
| `4096 x 512` | `warp_small_row` | `20.25599964` | `warp_small_row` | `1656.518197` |
| `4096 x 1024` | `warp_small_row` | `45.64800113` | `warp_small_row` | `1470.138064` |
| `1024 x 4096` | `warp_small_row` | `48.49600047` | `warp_small_row` | `1383.802032` |

In this run, `warp_small_row` was fastest and had the highest estimated
effective bandwidth for every measured shape. That agrees with the intended
small-row design: a warp can reduce a row with shuffle operations and avoid
block-wide synchronization. Its win at `1024 x 4096` is useful evidence for this
implementation, but it should not be generalized without repeated runs and
additional profiling at that large-row shape.

The one-thread-per-row `stable_two_pass` baseline is numerically stable but slow
because every row reduction is serial within one thread. `block_reduce` improves
large-row parallelism substantially; for `1024 x 4096`, its median latency was
`56.04799837 us` versus `917.6319838 us` for `stable_two_pass`.

Nsight reports under `results/rtx_5060_ti/ncu/` recorded long scoreboard as the
top captured stall reason for the profiled softmax variants. That supports the
expected model that memory dependency and row-wise reduction traffic matter,
while still keeping benchmark data and profiler data separate.
