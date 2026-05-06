# cuda-perf-lab

Architecture-aware CUDA kernel optimization experiments for AI inference and
numerical finance workloads.

The repo currently contains completed reduction and softmax v1 artifacts with
RTX 5060 Ti benchmark and Nsight evidence. Monte Carlo v1 benchmark code is
present with TODO result placeholders until real runs are generated. Matmul is
not implemented yet.

## What This Repo Proves

This repo is built to show kernel-level reasoning, not just timing output.

- Multiple implementation stages: naive to shared memory to warp shuffle to
  vectorized loads.
- Correctness against a CPU double-precision reference.
- Real CUDA event timing with warmup and timed iterations.
- CUB as the production-grade baseline.
- Roofline and Nsight Compute notes for explaining why a kernel is fast or slow.

## Highlights

- Reduction: the custom `vectorized_float4` kernel reached `1.006358415x` CUB
  latency at `n=67108864` in the recorded RTX 5060 Ti run, with repeated runs
  needed before treating that small gap as stable.
- Softmax: the warp-per-row implementation was fastest across the tested
  shapes, including `1024 x 4096`, but that result is implementation-specific
  rather than a universal softmax rule.
- Profiling: Nsight Compute reports for reduction and softmax point to memory
  dependency as a central bottleneck, with long scoreboard stalls appearing in
  the profiled kernels.
- Monte Carlo: Black-Scholes v1 adds parallel path simulation, cuRAND variants,
  CUB payoff reductions, standard error, and confidence intervals for a
  quant-oriented GPU workload. Result tables stay TODO until real runs are
  generated.

## Workflow

```text
kernel variants -> benchmark harness -> CSV/JSON results -> Nsight Compute reports -> README/docs interpretation
```

## Reproduce Results

From the repo root in WSL, regenerate the benchmark CSV/JSON results with:

```bash
cmake --preset configure-release
cmake --build --preset build-release
./scripts/run_reduction.sh && ./scripts/run_softmax.sh && ./scripts/run_monte_carlo.sh
```

Nsight Compute reports use the `profile_*_ncu.sh` scripts documented in the
profiling sections below because those runs are slower and profiler-specific.

## Hardware Target

Canonical environment:

```text
OS: WSL Ubuntu
GPU: NVIDIA GeForce RTX 5060 Ti
CUDA: 13.2
Build: CMake + Ninja + g++
Profiler: Nsight Compute CLI, ncu
```

The default CUDA architecture is `120` for the RTX 5060 Ti. Override it at
configure time if you build elsewhere:

```bash
cmake --preset configure-release -DCMAKE_CUDA_ARCHITECTURES=90
```

## Build

From the repo root in WSL:

```bash
cmake --preset configure-release
cmake --build --preset build-release
```

Debug build:

```bash
cmake --preset configure-debug
cmake --build --preset build-debug
```

The core benchmark does not require Python, PyTorch, or Triton.

## Run The Benchmark

Run all reduction variants on a small input:

```bash
./build/release/reduction_bench --variant all --n 1048576 --iters 5 --warmup 2 --seed 123
```

CSV output:

```bash
./build/release/reduction_bench --variant all --n 1048576 --iters 5 --warmup 2 --csv
```

JSON output:

```bash
./build/release/reduction_bench --variant warp_shuffle --n 1048576 --iters 5 --warmup 2 --json
```

Generate representative RTX 5060 Ti CSV and JSON results:

```bash
./scripts/run_reduction.sh
```

The script writes:

```text
results/rtx_5060_ti/reduction_results.csv
results/rtx_5060_ti/reduction_results.json
```

By default, it runs all variants for:

```text
1048576
16777216
67108864
```

with `--warmup 10 --iters 100`. You can override run parameters with
environment variables:

```bash
ITERS=20 WARMUP=5 SEED=123 ./scripts/run_reduction.sh
```

## Reduction Variants

| Variant | Purpose |
| --- | --- |
| `naive_global` | Bad baseline using repeated global-memory pairwise passes. |
| `shared_interleaved` | Classic shared-memory tree with interleaved addressing. |
| `shared_sequential` | Shared-memory tree with sequential addressing to reduce divergence. |
| `warp_shuffle` | Warp-level shuffle reduction plus shared warp partials. |
| `vectorized_float4` | First pass uses safe `float4` loads plus scalar tail handling. |
| `cub_baseline` | CUB `DeviceReduce::Sum` production baseline. |
| `all` | Runs every variant and reports ratio to CUB when available. |

## Benchmark Methodology

- Input is deterministic float32 data generated from `--seed`.
- CPU reference is a double-precision sum.
- GPU output is one float32 sum.
- Correctness reports absolute and relative error.
- Each variant runs warmup iterations before timed iterations.
- Timed iterations use CUDA events.
- Median and p95 latency are reported in microseconds.
- Bandwidth uses an estimated global-memory byte count for each variant:

```text
bandwidth_GB_s = estimated_global_bytes_moved / median_latency_seconds / 1e9
```

For custom staged kernels, the estimate includes global reads and writes for
each reduction pass plus the final device-to-device output copy. For CUB, the
estimate uses the minimum required traffic, input read plus one output write,
because CUB's internal traffic is implementation-dependent.

## Result Table

Source: `results/rtx_5060_ti/reduction_results.csv`.

| n | variant | median us | p95 us | GB/s | abs error | rel error | CUB ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1048576 | naive_global | 125.4880056 | 164.192006 | 100.2717984 | 0.01214754669 | 2.318380052e-08 | 10.37433917 |
| 1048576 | shared_interleaved | 27.11999975 | 49.44000021 | 155.2629808 | 0.01910245331 | 3.645735871e-08 | 2.242063489 |
| 1048576 | shared_sequential | 26.44800022 | 44.89599913 | 159.2079539 | 0.01214754669 | 2.318380052e-08 | 2.186507972 |
| 1048576 | warp_shuffle | 19.91999988 | 48.57600108 | 211.3821298 | 0.01910245331 | 3.645735871e-08 | 1.6468254 |
| 1048576 | vectorized_float4 | 19.76000052 | 47.71199822 | 212.6783345 | 0.01910245331 | 3.645735871e-08 | 1.63359794 |
| 1048576 | cub_baseline | 12.0959999 | 32.25599974 | 346.7516562 | 0.01214754669 | 2.318380052e-08 | 1 |
| 16777216 | naive_global | 458.7679952 | 490.5599952 | 438.8418331 | 0.2166625597 | 2.582164934e-08 | 2.743827615 |
| 16777216 | shared_interleaved | 196.1119995 | 199.7759938 | 343.5360008 | 0.2166625597 | 2.582164934e-08 | 1.172918611 |
| 16777216 | shared_sequential | 192.2239959 | 196.5440065 | 350.4845047 | 0.2166625597 | 2.582164934e-08 | 1.149665002 |
| 16777216 | warp_shuffle | 172.480002 | 184.5120043 | 390.6048887 | 0.7833374403 | 9.335745287e-08 | 1.031578919 |
| 16777216 | vectorized_float4 | 172.3679975 | 175.9359986 | 389.5253701 | 0.7833374403 | 9.335745287e-08 | 1.030909035 |
| 16777216 | cub_baseline | 167.2000065 | 182.4959964 | 401.368812 | 0.2166625597 | 2.582164934e-08 | 1 |
| 67108864 | naive_global | 1992.256045 | 2045.919895 | 404.2183061 | 1.155119803 | 3.442534423e-08 | 3.117030183 |
| 67108864 | shared_interleaved | 737.1839881 | 754.783988 | 365.5615102 | 1.155119803 | 3.442534423e-08 | 1.153378225 |
| 67108864 | shared_sequential | 721.1199999 | 730.783999 | 373.7049202 | 1.155119803 | 3.442534423e-08 | 1.128244941 |
| 67108864 | warp_shuffle | 646.592021 | 664.0319824 | 416.7791795 | 0.8448801972 | 2.517945892e-08 | 1.011640472 |
| 67108864 | vectorized_float4 | 643.2159841 | 660.3839993 | 417.3843726 | 0.8448801972 | 2.517945892e-08 | 1.006358415 |
| 67108864 | cub_baseline | 639.1519904 | 659.4240069 | 419.9868952 | 1.155119803 | 3.442534423e-08 | 1 |

Generate a README-ready table from real CSV output:

```bash
./scripts/summarize_reduction_results.py
```

The summarizer exits nonzero if the CSV is missing, so an empty table is not
accidentally treated as evidence.

The JSON file `results/rtx_5060_ti/reduction_results.json` is also present, but
the table above uses the CSV as its source of truth.

## How To Interpret Reduction Results

- `cub_baseline` was the fastest variant at all three measured sizes.
- `vectorized_float4` was the closest custom kernel to CUB in this run:
  `1.63359794x` CUB latency at `n=1048576`, `1.030909035x` at `n=16777216`,
  and `1.006358415x` at `n=67108864`. The largest-size gap is small enough
  that repeated runs are needed before drawing strong conclusions.
- `warp_shuffle` was also close to CUB at larger sizes: `1.031578919x` at
  `n=16777216` and `1.011640472x` at `n=67108864`.
- At the largest measured size, latency ratios versus CUB were:
  `naive_global` `3.117030183x`, `shared_interleaved` `1.153378225x`,
  `shared_sequential` `1.128244941x`, `warp_shuffle` `1.011640472x`, and
  `vectorized_float4` `1.006358415x`.
- The highest bandwidth column value at `n=1048576` was CUB
  (`346.7516562 GB/s`), at `n=16777216` was `naive_global`
  (`438.8418331 GB/s`), and at `n=67108864` was CUB (`419.9868952 GB/s`).
  The `naive_global` bandwidth should be read carefully because this benchmark
  reports estimated bytes moved; repeated global-memory passes can inflate that
  metric while still producing much worse latency.
- The measured behavior mostly matches the roofline expectation: larger
  reductions become bandwidth-dominated, and the optimized custom variants move
  toward CUB as launch overhead and inefficient reduction structure matter less.
  For small `n`, fixed launch overhead is still visible.

## Nsight Compute

Profile representative variants:

```bash
./scripts/profile_reduction_ncu.sh
```

Reports are saved under:

```text
results/rtx_5060_ti/ncu/
```

Captured reports:

```text
reduction_shared_sequential_n67108864_iters5.ncu-rep
reduction_warp_shuffle_n67108864_iters5.ncu-rep
reduction_vectorized_float4_n67108864_iters5.ncu-rep
reduction_cub_baseline_n67108864_iters5.ncu-rep
```

Summary from the primary large-grid kernel in each report:

| variant | profiled kernel | achieved occupancy | registers/thread | DRAM throughput | memory throughput | shared memory/block | global load efficiency | top captured stall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `shared_sequential` | `shared_sequential_kernel`, grid `(131072, 1, 1)` | `85.843911%` | `16` | `375728410454.803284 byte/s` | `87.923180%` | `2048 byte/block` | not captured | long scoreboard, `7.164042 inst` |
| `warp_shuffle` | `warp_shuffle_kernel`, grid `(131072, 1, 1)` | `79.418198%` | `16` | `425000113648.838074 byte/s` | `96.330052%` | `1152 byte/block` | not captured | long scoreboard, `34.610902 inst` |
| `vectorized_float4` | `vectorized_float4_kernel`, grid `(4096, 1, 1)` | `99.125517%` | `34` | `426088859449.335754 byte/s` | `96.575119%` | `1152 byte/block` | not captured | long scoreboard, `362.867479 inst` |
| `cub_baseline` | `DeviceReduceKernel`, grid `(1080, 1, 1)` | `99.000641%` | `40` | `425715087478.788025 byte/s` | `96.491978%` | `1072 byte/block` | not captured | long scoreboard, `554.136743 inst` |

Nsight also captured `32 byte/sector` average global-load data bytes per sector
for those primary kernels, but it did not capture a metric named global load
efficiency.

The profiling script defaults to `N=67108864` and profiles:

```text
shared_sequential
warp_shuffle
vectorized_float4
cub_baseline
```

Metric names can vary across Nsight Compute releases and GPU architectures. The
script defaults to `--set full`. To force specific metrics:

```bash
NCU_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed" ./scripts/profile_reduction_ncu.sh
```

## What To Look For In Nsight Compute

- Achieved occupancy.
- Register count.
- Memory throughput.
- Global load efficiency.
- Shared memory usage.
- Warp stall reasons.
- Achieved FLOP/s, mostly to confirm reduction is not compute-bound.

Record findings in `docs/nsight_analysis.md`. Do not paste guessed metrics.

## Softmax

Softmax v1 adds an AI inference workload: row-wise float32 softmax over a
row-major matrix. It proves max reduction, exponential normalization, sum
reduction, and row-wise memory traffic behavior without requiring PyTorch or
Triton.

### Why Softmax Matters

Softmax appears in attention and classification-style inference paths. A good
softmax kernel is not just "take exp"; it has to reduce each row for the max,
reduce again for the denominator, and write normalized probabilities with
controlled memory traffic.

### Softmax Variants

| Variant | Purpose |
| --- | --- |
| `naive` | One-thread-per-row baseline using bounded inputs. |
| `stable_two_pass` | Stable one-thread-per-row max/sum/normalize baseline. |
| `block_reduce` | One CUDA block per row for moderate and large rows. |
| `warp_small_row` | One warp per row with lane-strided loops; optimized for small rows and safe for larger rows. |
| `all` | Runs every softmax variant and reports ratio to `stable_two_pass`. |

Despite the name, `warp_small_row` has no hard small-row correctness limit.
Each lane walks `col = lane; col < cols; col += warpSize` for the max, sum,
and output passes. The name describes the intended design target: row widths
where one warp per row has low overhead and avoids block-wide synchronization.

### Run Softmax

Build and run a small correctness smoke test:

```bash
cmake --build --preset build-release
./build/release/softmax_bench --variant all --rows 16 --cols 128 --iters 3 --warmup 1 --seed 123
```

Generate representative RTX 5060 Ti softmax CSV and JSON results:

```bash
./scripts/run_softmax.sh
```

The script writes:

```text
results/rtx_5060_ti/softmax_results.csv
results/rtx_5060_ti/softmax_results.json
```

### Softmax Methodology

- Input is deterministic bounded float32 data generated from `--seed`.
- CPU reference uses double precision and numerically stable softmax.
- Correctness reports max absolute error, max relative error, and max row-sum
  error.
- Warmups are excluded from CUDA-event timing.
- Bandwidth is estimated from documented global matrix traffic.
- No GFLOP/s is reported in v1 because the `expf` cost model is not simple
  enough for an honest one-line estimate.
- Baseline ratio is relative to `stable_two_pass`.

### Softmax Results

Source: `results/rtx_5060_ti/softmax_results.csv`. Benchmark data and Nsight
profiler data are separate; the table below is benchmark data only.

| rows | cols | variant | median us | p95 us | GB/s | max abs error | max rel error | row sum error | baseline ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 128 | naive | 29.05599959 | 45.50400004 | 216.5286374 | 4.261725935e-08 | 7.736328632e-07 | 6.573936844e-07 | 0.8024745792 |
| 4096 | 128 | stable_two_pass | 36.20800003 | 52.41600052 | 231.6783029 | 4.467263601e-08 | 9.484457293e-07 | 6.073441909e-07 | 1 |
| 4096 | 128 | block_reduce | 47.64800146 | 50.65599829 | 176.0537219 | 1.33545618e-08 | 4.924956992e-07 | 1.602293196e-07 | 1.315952315 |
| 4096 | 128 | warp_small_row | 10.36799978 | 28.92800048 | 809.0864366 | 1.358760054e-08 | 4.924956992e-07 | 1.412863639e-07 | 0.2863455528 |
| 4096 | 512 | naive | 100.6079987 | 109.0560034 | 250.1374079 | 2.080695207e-08 | 1.279423175e-06 | 1.140637778e-06 | 0.804709448 |
| 4096 | 512 | stable_two_pass | 125.0240058 | 141.7600065 | 268.3839139 | 1.974259728e-08 | 1.545079688e-06 | 1.216165401e-06 | 1 |
| 4096 | 512 | block_reduce | 53.44000086 | 55.23199961 | 627.8898102 | 3.340429246e-09 | 5.098644972e-07 | 1.52343091e-07 | 0.4274379191 |
| 4096 | 512 | warp_small_row | 20.25599964 | 32.03200176 | 1656.518197 | 3.855019576e-09 | 5.397656429e-07 | 1.75479272e-07 | 0.1620168824 |
| 4096 | 1024 | naive | 193.2479963 | 196.352005 | 260.4510731 | 1.421844931e-08 | 1.83575157e-06 | 1.689710416e-06 | 0.7944484464 |
| 4096 | 1024 | stable_two_pass | 243.2480007 | 247.5520074 | 275.8866005 | 1.221485248e-08 | 1.841952763e-06 | 1.482567313e-06 | 1 |
| 4096 | 1024 | block_reduce | 66.46399945 | 86.65599674 | 1009.702464 | 1.590226388e-09 | 5.143566691e-07 | 1.588673513e-07 | 0.2732355425 |
| 4096 | 1024 | warp_small_row | 45.64800113 | 64.80000168 | 1470.138064 | 1.958800033e-09 | 5.477278315e-07 | 2.019933163e-07 | 0.1876603343 |
| 1024 | 4096 | naive | 730.1440239 | 902.8159976 | 68.93386284 | 7.043033519e-09 | 3.837401178e-06 | 3.658493938e-06 | 0.7956828411 |
| 1024 | 4096 | stable_two_pass | 917.6319838 | 1100.736022 | 73.13265578 | 5.982558308e-09 | 3.329482876e-06 | 2.962015401e-06 | 1 |
| 1024 | 4096 | block_reduce | 56.04799837 | 62.97600269 | 1197.346309 | 4.303151923e-10 | 5.249813505e-07 | 1.337249387e-07 | 0.06107895034 |
| 1024 | 4096 | warp_small_row | 48.49600047 | 64.76800144 | 1383.802032 | 4.911223503e-10 | 5.511743613e-07 | 1.634575142e-07 | 0.05284907384 |

In this benchmark run, `warp_small_row` was the fastest and had the highest
estimated effective bandwidth for every measured shape. That matches the
expected benefit of warp-level row reductions for small and moderate rows, and
it also won the `1024 x 4096` large-row shape in this implementation. The
large-row case is still handled by the same lane-strided loops described above,
so it is safe but not necessarily the best policy on every GPU, row width, or
implementation. Treat that result as implementation-specific, not as a universal
claim that one-warp-per-row softmax always beats block-per-row softmax.

### Softmax Nsight Compute

Profile representative softmax variants:

```bash
./scripts/profile_softmax_ncu.sh
```

Reports are saved under:

```text
results/rtx_5060_ti/ncu/
```

Look for achieved occupancy, register count, memory throughput, global load
behavior, shared memory usage, and warp stall reasons. Missing metrics should be
recorded as `not captured`.

Captured softmax reports:

```text
softmax_stable_two_pass_rows4096_cols512_iters5.ncu-rep
softmax_block_reduce_rows1024_cols4096_iters5.ncu-rep
softmax_warp_small_row_rows4096_cols128_iters5.ncu-rep
```

Summary from the profiled kernel in each report:

| variant | shape | achieved occupancy | registers/thread | DRAM throughput | memory throughput | global load behavior | shared memory/block | top captured stall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stable_two_pass` | `4096 x 512` | `8.321258%` | `39` | `99397856793.850861 byte/s` | `69.269273%` | `4 byte/sector` | `1024 byte/block` | long scoreboard, `13.171933 inst` |
| `block_reduce` | `1024 x 4096` | `91.832641%` | `22` | `308736740135.620300 byte/s` | `70.005227%` | `32 byte/sector` | `2048 byte/block` | long scoreboard, `20.942470 inst` |
| `warp_small_row` | `4096 x 128` | `77.515592%` | `40` | `271105929108.687744 byte/s` | `61.676348%` | `32 byte/sector` | `1024 byte/block` | long scoreboard, `8.019646 inst` |

The profiler data is from targeted shapes, not the full benchmark matrix. It
supports the expected model that global-memory dependency and row reductions are
important: all three profiled variants report long scoreboard as the top
captured stall reason.

### Why Framework Softmax May Still Win

PyTorch, Triton, and vendor/library softmax kernels can fuse operations, tune
for specific row widths, use architecture-specific launch policy, and avoid
extra reads or writes in ways this educational v1 does not. Beating them is not
the claim; explaining the gap is the point.

## Monte Carlo

Monte Carlo v1 adds a numerical finance workload: Black-Scholes European call
option pricing with GPU path simulation. It demonstrates random number
generation cost, payoff generation, payoff reduction, statistical error,
confidence intervals, and latency/throughput tradeoffs.

### Why Monte Carlo Matters

Monte Carlo is common in quant development because many pricing and risk
problems are easier to simulate than solve analytically. It is also a useful GPU
performance workload because independent paths expose parallelism, while RNG,
`expf`, and payoff reductions keep it from being a pure memory-copy benchmark.

The benchmark reports a double-precision CPU analytical Black-Scholes price as
correctness context. Monte Carlo estimates vary with random draws, so the
benchmark reports standard error and a 95 percent confidence interval instead
of requiring exact equality.

### Monte Carlo Variants

| Variant | Purpose |
| --- | --- |
| `cpu_baseline` | Double-precision analytical Black-Scholes reference only. |
| `gpu_naive_curand` | One thread per path using XORWOW cuRAND state. |
| `gpu_philox` | One thread per path using Philox cuRAND state; latency ratio baseline. |
| `gpu_philox_antithetic` | Philox path simulation with antithetic `z` and `-z` payoff averaging. |
| `all` | Runs every Monte Carlo variant. |

### Run Monte Carlo

Build and run a small smoke test:

```bash
cmake --build --preset build-release
./build/release/monte_carlo_bench --variant all --paths 1024 --steps 1 --iters 3 --warmup 1 --seed 123
```

Generate representative RTX 5060 Ti Monte Carlo CSV and JSON results:

```bash
./scripts/run_monte_carlo.sh
```

The script writes:

```text
results/rtx_5060_ti/monte_carlo_results.csv
results/rtx_5060_ti/monte_carlo_results.json
```

### Monte Carlo Methodology

- Default option parameters are `spot=100`, `strike=100`, `rate=0.05`,
  `vol=0.2`, and `maturity=1.0`.
- For `steps=1`, GPU kernels sample the terminal geometric Brownian motion
  distribution directly.
- For `steps>1`, GPU kernels use log-space Euler stepping.
- GPU simulation is float32; the analytical Black-Scholes reference is double
  precision on CPU.
- GPU timing includes random draw/path simulation, payoff generation, payoff
  reduction, and payoff-squared reduction.
- GPU timing excludes memory allocation, RNG state initialization, CUB temp
  allocation, host-side result copies, and formatting.
- CUB `DeviceReduce::Sum` is used for payoff and payoff-squared reductions.
- `baseline_ratio` is relative to `gpu_philox`; `gpu_philox` reports `1.0`,
  and `cpu_baseline` leaves the ratio empty/null.

### Monte Carlo Results

Source: `results/rtx_5060_ti/monte_carlo_results.csv`.

TODO: run `./scripts/run_monte_carlo.sh` on the RTX 5060 Ti and paste only
real generated rows here.

| paths | steps | variant | median us | p95 us | paths/s | estimated price | analytical price | standard error | 95% CI | baseline ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### Monte Carlo Nsight Compute

Profile representative Monte Carlo variants:

```bash
./scripts/profile_monte_carlo_ncu.sh
```

Reports are saved under:

```text
results/rtx_5060_ti/ncu/
```

Captured report summary:

TODO: fill this table only after real `.ncu-rep` files are generated and
inspected. Missing metrics should be recorded as `not captured`.

| variant | paths | steps | achieved occupancy | registers/thread | memory throughput | global load/store behavior | top captured stall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## Smoke Verification

These commands were used for smoke testing during v1 development:

```bash
cmake --build --preset build-release
./build/release/reduction_bench --variant all --n 1048576 --iters 5 --warmup 2 --seed 123
./build/release/reduction_bench --variant vectorized_float4 --n 1048581 --iters 3 --warmup 1 --seed 123
./build/release/monte_carlo_bench --variant all --paths 1024 --steps 1 --iters 3 --warmup 1 --seed 123
./build/release/monte_carlo_bench --variant all --paths 4096 --steps 16 --iters 3 --warmup 1 --seed 123
```

They are correctness and CLI checks, not final benchmark evidence.

## Why My Kernel Probably Does Not Beat CUB

CUB is not a toy baseline. It has tuned policies, architecture-specific choices,
careful grid sizing, efficient memory access, and mature handling for many input
sizes. A local educational kernel can be useful without beating CUB.

The goal is to explain the gap:

- Did my kernel launch too many passes?
- Did it move extra global memory traffic?
- Did it leave memory bandwidth unused?
- Did register pressure or occupancy limit parallelism?
- Did CUB use a better load, reduce, or writeback policy?

That failure analysis is part of the point.

## Planned Work

- Matmul: tiled matrix multiplication with cuBLAS comparison.

## Repo Layout

```text
README.md
AGENTS.md
CMakeLists.txt
CMakePresets.json
include/
  common/
kernels/
  reduction/
  softmax/
  monte_carlo/
benchmarks/
scripts/
docs/
results/
  rtx_5060_ti/
```
