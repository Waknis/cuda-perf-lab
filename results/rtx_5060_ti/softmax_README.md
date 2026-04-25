# Softmax Results

This directory will hold softmax benchmark and profiling artifacts produced on
the RTX 5060 Ti.

Expected generated files:

```text
softmax_results.csv
softmax_results.json
ncu/softmax_*.ncu-rep
```

Generate benchmark results with:

```bash
./scripts/run_softmax.sh
```

Generate Nsight Compute reports with:

```bash
./scripts/profile_softmax_ncu.sh
```

Do not hand-edit benchmark or profiling numbers into this directory.
