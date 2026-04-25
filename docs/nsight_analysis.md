# Nsight Compute Analysis

Do not fill this file with guessed numbers. Values below come from files under
`results/rtx_5060_ti/`.

Benchmark latency and effective bandwidth come from
`results/rtx_5060_ti/reduction_results.csv`. Profiling counters come from the
Nsight Compute `.ncu-rep` files in `results/rtx_5060_ti/ncu/`.

## Hardware

```text
GPU: NVIDIA GeForce RTX 5060 Ti
CUDA: 13.2
Driver: 595.97
Commit: unknown
Profiler command: scripts/profile_reduction_ncu.sh
Profiler command line in reports: ncu --target-processes all --force-overwrite --set full --export ... reduction_bench --variant <variant> --n 67108864 --iters 5 --warmup 2 --seed 1234
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

# Softmax Nsight Templates

Do not fill these with guessed values. Use only `softmax_*.ncu-rep` files under
`results/rtx_5060_ti/ncu/`. If a metric is absent, write `not captured`.

### stable_two_pass

```text
input shape: TODO
median benchmark latency: TODO
benchmark bandwidth: TODO
baseline ratio: TODO
profiled kernel: TODO
achieved occupancy: not captured
register count: not captured
memory throughput: not captured
global load behavior: not captured
shared memory usage: not captured
top warp stall reason: not captured
interpretation: TODO
next experiment: TODO
```

### block_reduce

```text
input shape: TODO
median benchmark latency: TODO
benchmark bandwidth: TODO
baseline ratio: TODO
profiled kernel: TODO
achieved occupancy: not captured
register count: not captured
memory throughput: not captured
global load behavior: not captured
shared memory usage: not captured
top warp stall reason: not captured
interpretation: TODO
next experiment: TODO
```

### warp_small_row

```text
input shape: TODO
median benchmark latency: TODO
benchmark bandwidth: TODO
baseline ratio: TODO
profiled kernel: TODO
achieved occupancy: not captured
register count: not captured
memory throughput: not captured
global load behavior: not captured
shared memory usage: not captured
top warp stall reason: not captured
interpretation: TODO
next experiment: TODO
```
