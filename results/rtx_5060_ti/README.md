# RTX 5060 Ti Results

This directory is for benchmark and profiling artifacts produced on:

```text
GPU: NVIDIA GeForce RTX 5060 Ti
Environment: WSL Ubuntu
CUDA: 13.2
```

Do not hand-edit benchmark numbers into this directory.

Expected generated files:

```text
reduction_results.csv
ncu/
```

Generate reduction benchmark results with:

```bash
./scripts/run_reduction.sh
```

Generate Nsight Compute reports with:

```bash
./scripts/profile_reduction_ncu.sh
```
