# Nsight Compute Analysis

Do not fill this file with guessed numbers. Values below come from files under
`results/rtx_5060_ti/`.

Benchmark latency and effective bandwidth come from
`results/rtx_5060_ti/reduction_results.csv` and
`results/rtx_5060_ti/softmax_results.csv`. Profiling counters come from the
Nsight Compute `.ncu-rep` files in `results/rtx_5060_ti/ncu/`.

## Hardware

```text
GPU: NVIDIA GeForce RTX 5060 Ti
CUDA: 13.2
Driver: 595.97
Commit: unknown
Reduction profiler command: scripts/profile_reduction_ncu.sh
Softmax profiler command: scripts/profile_softmax_ncu.sh
Reduction profiler command line in reports: ncu --target-processes all --force-overwrite --set full --export ... reduction_bench --variant <variant> --n 67108864 --iters 5 --warmup 2 --seed 1234
Nsight Compute version: 2026.1.1.0 (build 37634170)
```

## Metrics To Record

```text
Variant:
Input n:
Median benchmark latency:

Achieved occupancy:
Register count:
Memory throughput:
Global load efficiency:
Shared memory usage:
Achieved FLOP/s:
Top warp stall reasons:
Notes:
```

## Filled Templates

The Nsight values below use the primary large-grid kernel in each report. Some
variants launch helper/final reduction kernels too; those are captured in the
`.ncu-rep` files but are not the main row summarized here.

### shared_sequential

```text
input n: 67108864
median benchmark latency: 721.1199999 us
benchmark bandwidth: 373.7049202 GB/s
baseline ratio: 1.128244941
profiled kernel: shared_sequential_kernel, grid (131072, 1, 1)
achieved occupancy: 85.843911%
register count: 16 registers/thread
memory throughput: 375728410454.803284 byte/s DRAM, 87.923180% peak sustained elapsed
global load efficiency: not captured
global load bytes/sector: 32 byte/sector
shared memory usage: 2048 byte/block total, 1024 byte/block dynamic, 0 byte/block static
top warp stall reason: long scoreboard, 7.164042 inst
interpretation: Sequential addressing improves over the interleaved shared-memory version in the CSV run, but the primary launch reaches lower DRAM throughput than warp_shuffle, vectorized_float4, and CUB.
next experiment: Compare long-scoreboard stalls and DRAM throughput against warp_shuffle to isolate whether memory dependency or reduction structure is the larger gap.
```

### warp_shuffle

```text
input n: 67108864
median benchmark latency: 646.592021 us
benchmark bandwidth: 416.7791795 GB/s
baseline ratio: 1.011640472
profiled kernel: warp_shuffle_kernel, grid (131072, 1, 1)
achieved occupancy: 79.418198%
register count: 16 registers/thread
memory throughput: 425000113648.838074 byte/s DRAM, 96.330052% peak sustained elapsed
global load efficiency: not captured
global load bytes/sector: 32 byte/sector
shared memory usage: 1152 byte/block total, 0 byte/block dynamic, 128 byte/block static
top warp stall reason: long scoreboard, 34.610902 inst
interpretation: The primary warp-shuffle launch reaches high memory throughput and is close to CUB in the CSV benchmark, consistent with a bandwidth-bound large reduction.
next experiment: Compare helper reduction launches and stall mix against vectorized_float4, since the large primary launch is already near CUB memory throughput.
```

### vectorized_float4

```text
input n: 67108864
median benchmark latency: 643.2159841 us
benchmark bandwidth: 417.3843726 GB/s
baseline ratio: 1.006358415
profiled kernel: vectorized_float4_kernel, grid (4096, 1, 1)
achieved occupancy: 99.125517%
register count: 34 registers/thread
memory throughput: 426088859449.335754 byte/s DRAM, 96.575119% peak sustained elapsed
global load efficiency: not captured
global load bytes/sector: 32 byte/sector
shared memory usage: 1152 byte/block total, 0 byte/block dynamic, 128 byte/block static
top warp stall reason: long scoreboard, 362.867479 inst
interpretation: This was the closest custom variant to CUB at the large size in this CSV run, and the primary vectorized load kernel reports similar DRAM throughput to CUB. The 1.006358415x benchmark gap is small enough that repeated runs are needed before drawing strong conclusions.
next experiment: Repeat full benchmark runs and inspect helper warp_shuffle launches to see whether the small CUB gap is stable or run-dependent.
```

### cub_baseline

```text
input n: 67108864
median benchmark latency: 639.1519904 us
benchmark bandwidth: 419.9868952 GB/s
baseline ratio: 1
profiled kernel: DeviceReduceKernel, grid (1080, 1, 1)
achieved occupancy: 99.000641%
register count: 40 registers/thread
memory throughput: 425715087478.788025 byte/s DRAM, 96.491978% peak sustained elapsed
global load efficiency: not captured
global load bytes/sector: 32 byte/sector
shared memory usage: 1072 byte/block total, 0 byte/block dynamic, 48 byte/block static
top warp stall reason: long scoreboard, 554.136743 inst
interpretation: CUB was fastest in the CSV run and its primary reduction kernel reaches high occupancy and high DRAM throughput. The report also contains DeviceReduceSingleTileKernel for the final tile.
next experiment: Compare CUB's two-kernel policy with the custom vectorized first pass plus warp-shuffle cleanup.
```

## Benchmark vs Profiler Evidence

- Fastest CSV variant at `n=67108864`: `cub_baseline`, `639.1519904 us`.
- Highest estimated effective bandwidth at `n=67108864`: `cub_baseline`,
  `419.9868952 GB/s`.
- Closest custom kernel to CUB at `n=67108864`: `vectorized_float4`,
  `1.006358415x` CUB latency. This is a small gap and should be treated as
  run-dependent unless repeated runs confirm it.
- The profiler evidence matches the memory-bound expectation: the primary
  large-load kernels for `warp_shuffle`, `vectorized_float4`, and CUB all report
  roughly `96%` memory throughput, while the top captured stall reason is long
  scoreboard.

## What To Look For

- Whether optimized variants move closer to peak memory throughput.
- Whether warp-shuffle reduction lowers shared-memory instructions.
- Whether register count changes enough to affect occupancy.
- Whether stalls are dominated by memory dependency, memory throttle, or launch
  overhead.
- Whether CUB uses a different launch strategy or fewer bottleneck symptoms.

## Softmax Nsight Evidence

The softmax values below use only `softmax_results.csv` and the
`softmax_*.ncu-rep` files under `results/rtx_5060_ti/ncu/`. Benchmark rows and
Nsight profiler rows are related by shape, but they are separate measurements.
If a metric was absent from the extracted report data, it is written as
`not captured`.

### stable_two_pass

```text
input shape: 4096 x 512
median benchmark latency: 125.0240058 us
benchmark bandwidth: 268.3839139 GB/s
baseline ratio: 1
profiled kernel: stable_two_pass_kernel, grid (32, 1, 1)
achieved occupancy: 8.321258%
register count: 39 registers/thread
memory throughput: 99397856793.850861 byte/s DRAM, 69.269273% peak sustained elapsed
global load behavior: 4 byte/sector
shared memory usage: 1024 byte/block total, 0 byte/block dynamic, 0 byte/block static
top warp stall reason: long scoreboard, 13.171933 inst
interpretation: The one-thread-per-row stable baseline has low achieved occupancy and scalar-looking global load behavior in the profiled shape. It is useful as a correctness and ratio baseline, but the CSV benchmark shows much higher latency than the cooperative variants.
next experiment: Compare against a multi-row-per-block scalar baseline to separate poor occupancy from the cost of serial row reductions.
```

### block_reduce

```text
input shape: 1024 x 4096
median benchmark latency: 56.04799837 us
benchmark bandwidth: 1197.346309 GB/s
baseline ratio: 0.06107895034
profiled kernel: block_reduce_kernel, grid (1024, 1, 1)
achieved occupancy: 91.832641%
register count: 22 registers/thread
memory throughput: 308736740135.620300 byte/s DRAM, 70.005227% peak sustained elapsed
global load behavior: 32 byte/sector
shared memory usage: 2048 byte/block total, 1024 byte/block dynamic, 0 byte/block static
top warp stall reason: long scoreboard, 20.942470 inst
interpretation: The block-per-row kernel uses cooperative reductions and reaches high occupancy on the profiled large-row shape. It is much faster than the serial stable baseline in the CSV data, but still trails warp_small_row for this measured implementation and shape.
next experiment: Sweep block size and rows-per-block policy for large rows to see whether better occupancy, fewer synchronizations, or more vectorized memory access changes the gap.
```

### warp_small_row

```text
input shape: 4096 x 128
median benchmark latency: 10.36799978 us
benchmark bandwidth: 809.0864366 GB/s
baseline ratio: 0.2863455528
profiled kernel: warp_small_row_kernel, grid (512, 1, 1)
achieved occupancy: 77.515592%
register count: 40 registers/thread
memory throughput: 271105929108.687744 byte/s DRAM, 61.676348% peak sustained elapsed
global load behavior: 32 byte/sector
shared memory usage: 1024 byte/block total, 0 byte/block dynamic, 0 byte/block static
top warp stall reason: long scoreboard, 8.019646 inst
interpretation: The warp-per-row kernel is the fastest softmax variant in the benchmark CSV for every measured shape and is especially aligned with the profiled small-row case. The large-row win in the CSV should be treated as implementation-specific, not a universal softmax rule.
next experiment: Repeat benchmark runs and profile warp_small_row on the 1024 x 4096 shape to confirm whether the large-row result is stable.
```

## Softmax Benchmark vs Profiler Evidence

- Fastest CSV variant for all four measured shapes: `warp_small_row`.
- Highest estimated effective bandwidth for all four measured shapes:
  `warp_small_row`.
- Small-row behavior matches the expected model: the warp-per-row path avoids
  block-wide synchronization and performs row reductions with shuffle
  instructions.
- The measured large-row behavior also favors `warp_small_row` in this run, but
  that should be treated as implementation-specific until repeated runs and
  additional profiling confirm it.
- The profiled softmax variants all report long scoreboard as the top captured
  stall reason, which is consistent with memory dependency being important in
  row-wise softmax.

## Monte Carlo Nsight Template

Monte Carlo profiler rows should be filled only from
`results/rtx_5060_ti/ncu/monte_carlo_*.ncu-rep`. If a metric is absent from the
report, write `not captured`.

### gpu_naive_curand

```text
input paths: TODO
input steps: TODO
median benchmark latency: TODO
paths per second: TODO
profiled kernel: TODO
achieved occupancy: TODO
register count: TODO
memory throughput: TODO
global load/store behavior: TODO
shared memory usage: TODO
warp stall reasons: TODO
special function or math pipeline notes: TODO
interpretation: TODO
next experiment: TODO
```

### gpu_philox

```text
input paths: TODO
input steps: TODO
median benchmark latency: TODO
paths per second: TODO
profiled kernel: TODO
achieved occupancy: TODO
register count: TODO
memory throughput: TODO
global load/store behavior: TODO
shared memory usage: TODO
warp stall reasons: TODO
special function or math pipeline notes: TODO
interpretation: TODO
next experiment: TODO
```

### gpu_philox_antithetic

```text
input paths: TODO
input steps: TODO
median benchmark latency: TODO
paths per second: TODO
profiled kernel: TODO
achieved occupancy: TODO
register count: TODO
memory throughput: TODO
global load/store behavior: TODO
shared memory usage: TODO
warp stall reasons: TODO
special function or math pipeline notes: TODO
interpretation: TODO
next experiment: TODO
```
