# Reduction Optimization Notes

This project keeps the reduction stages explicit on purpose. The goal is not a
zoo of kernels; it is a controlled path from bad baseline to credible baseline.

## naive_global

The naive variant repeatedly halves the array using global memory. It is a bad
baseline by design: each pass launches another kernel and writes intermediate
partials to global memory. It demonstrates why reducing only through global
memory wastes bandwidth and launch time.

## shared_interleaved

This is the classic shared-memory tree with interleaved addressing. It reduces
global memory traffic by creating one partial sum per block, but the modulo-based
thread condition creates divergence and awkward access patterns. It exists to
make the cost of a textbook-but-rough reduction visible.

## shared_sequential

Sequential addressing keeps active threads contiguous as the tree shrinks. That
removes the modulo branch pattern and is the first cleanup step that maps better
to how warps execute.

## warp_shuffle

Warp shuffle instructions reduce values within a warp without bouncing every
step through shared memory. This variant still uses shared memory for warp
partials, but the intra-warp part is register-to-register.

## vectorized_float4

The vectorized variant loads four contiguous float32 values at a time when it is
safe to do so. That can reduce load instruction overhead and improve memory
coalescing behavior for the first pass. A scalar tail path keeps correctness for
any `n`.

## cub_baseline

CUB is the library baseline. It is included so this repo compares against a real
production-grade primitive, not just against worse local kernels. If the custom
kernels do not beat CUB, that is expected and useful evidence.

# Softmax Variants

## naive

The naive softmax assigns one thread to each row and computes direct exponentials
without subtracting the row maximum. Inputs are bounded for v1 so this remains a
valid baseline, but it is not the numerically robust production form.

## stable_two_pass

This variant still uses one thread per row, but it computes a row max, then a
sum of shifted exponentials, then normalized output. It is simple and stable,
and serves as the custom baseline for softmax ratios.

## block_reduce

One CUDA block handles one row. Threads cooperate on the max reduction and sum
reduction, then write normalized output. This demonstrates block-level
reductions and synchronization for moderate or large row widths.

## warp_small_row

One warp handles one row using shuffle reductions for max and sum. This avoids
block-wide synchronization for small rows and is intended for `cols <= 1024`,
while remaining correct for larger rows by striding each lane across the row.

## Softmax Evidence Notes

Source: `results/rtx_5060_ti/softmax_results.csv`.

In the measured RTX 5060 Ti run, `warp_small_row` was the fastest softmax
variant for all benchmarked shapes:

| shape | `warp_small_row` median us | next fastest variant | next fastest median us |
| --- | --- | --- | --- |
| `4096 x 128` | `10.36799978` | `naive` | `29.05599959` |
| `4096 x 512` | `20.25599964` | `block_reduce` | `53.44000086` |
| `4096 x 1024` | `45.64800113` | `block_reduce` | `66.46399945` |
| `1024 x 4096` | `48.49600047` | `block_reduce` | `56.04799837` |

The small-row wins match the intended purpose of `warp_small_row`. The
`1024 x 4096` result shows that this implementation's warp-strided large-row
path was also strong in this run, but that should be treated as run-dependent
until repeated measurements confirm it.

Source: `results/rtx_5060_ti/ncu/softmax_*.ncu-rep`.

The profiled softmax reports captured long scoreboard as the top warp stall
reason for `stable_two_pass`, `block_reduce`, and `warp_small_row`. That is
consistent with memory dependency being an important limiter, but it does not
replace the benchmark CSV for latency comparisons.

# Monte Carlo Variants

## cpu_baseline

The CPU baseline computes the double-precision analytical Black-Scholes call
price. It exists for correctness context and does not define the GPU latency
ratio baseline.

## gpu_naive_curand

One GPU thread simulates one path using XORWOW cuRAND state. This is the simple
GPU baseline: path simulation, payoff generation, and payoff writes are direct
and readable.

## gpu_philox

One GPU thread simulates one path using Philox cuRAND state. Philox is the v1
latency-ratio baseline because it is deterministic and parallel-friendly.

## gpu_philox_antithetic

This variant uses each random draw together with its negation and averages the
paired payoffs. The purpose is to show variance/runtime tradeoffs, not to assume
variance reduction without measured standard-error output.

## CUB payoff reduction

Monte Carlo v1 uses CUB `DeviceReduce::Sum` for payoff and payoff-squared
arrays. That keeps the final aggregation credible and makes the benchmark focus
on the cost of simulation plus production-grade reductions.

## Future Monte Carlo Extensions

- Asian option.
- Barrier option.
- Quasi-random Sobol draws.
- Mixed precision simulation.
- Multi-asset simulation.
