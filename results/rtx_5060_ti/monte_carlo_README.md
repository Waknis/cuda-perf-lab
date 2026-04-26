# Monte Carlo Results

This directory will hold Monte Carlo benchmark and profiling artifacts produced
on the RTX 5060 Ti.

Expected generated files:

```text
monte_carlo_results.csv
monte_carlo_results.json
ncu/monte_carlo_*.ncu-rep
```

Generate benchmark results with:

```bash
./scripts/run_monte_carlo.sh
```

Generate Nsight Compute reports with:

```bash
./scripts/profile_monte_carlo_ncu.sh
```

Do not hand-edit benchmark or profiling numbers into this directory.
