# cuda-perf-lab

Architecture-aware CUDA kernel optimization experiments for AI inference and
numerical finance workloads.

V1 is intentionally narrow: it implements a staged float32 reduction benchmark
and compares the custom kernels against CUB. Softmax, matmul, and Monte Carlo
are planned, but not implemented yet.

## What This Repo Proves

This repo is built to show kernel-level reasoning, not just timing output.

- Multiple implementation stages: naive to shared memory to warp shuffle to
  vectorized loads.
- Correctness against a CPU double-precision reference.
- Real CUDA event timing with warmup and timed iterations.
- CUB as the production-grade baseline.
- Roofline and Nsight Compute notes for explaining why a kernel is fast or slow.

No benchmark number, Nsight metric, hardware result, or speedup in this repo
should be invented. Empty tables stay TODO until a command actually produces
data.

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

## Smoke Verification

These commands were used for smoke testing during v1 development:

```bash
cmake --build --preset build-release
./build/release/reduction_bench --variant all --n 1048576 --iters 5 --warmup 2 --seed 123
./build/release/reduction_bench --variant vectorized_float4 --n 1048581 --iters 3 --warmup 1 --seed 123
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

- V2: softmax kernels for AI inference relevance.
- V3: tiled matrix multiplication with cuBLAS comparison.
- V4: Monte Carlo option pricing for quant workloads.

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
benchmarks/
scripts/
docs/
results/
  rtx_5060_ti/
```
