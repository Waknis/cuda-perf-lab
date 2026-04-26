# Monte Carlo Notes

Monte Carlo v1 prices a European call option under geometric Brownian motion.
The option payoff is:

```text
max(S_T - K, 0)
```

The discounted Monte Carlo price is:

```text
exp(-rT) * average_payoff
```

## Black-Scholes Reference

The benchmark reports a double-precision CPU Black-Scholes analytical call
price. That value is correctness context, not a simulation baseline. Monte
Carlo estimates are random estimates, so exact equality is not expected.

## Path Simulation

For `steps = 1`, the benchmark samples the terminal distribution directly:

```text
S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
```

For `steps > 1`, it uses log-space Euler stepping:

```text
dt = T / steps
log S += (r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
```

The GPU v1 simulation uses float32. The analytical Black-Scholes reference uses
double precision on the CPU.

## Variants

- `cpu_baseline`: analytical Black-Scholes price only.
- `gpu_naive_curand`: one thread simulates one path with XORWOW cuRAND state.
- `gpu_philox`: one thread simulates one path with Philox cuRAND state.
- `gpu_philox_antithetic`: one thread simulates an antithetic pair and averages
  the paired payoffs.

## Statistical Error

The benchmark writes both payoff and payoff-squared arrays, then uses CUB
`DeviceReduce::Sum` to compute:

```text
sum_payoff
sum_payoff_squared
```

Those sums produce the mean payoff, sample variance, standard error, and a
95 percent confidence interval. If the analytical Black-Scholes price lands
outside the interval, the benchmark prints a warning but does not fail.

## Why RNG Can Dominate

This workload is not just a memory bandwidth test. Each path requires random
normal generation, exponentials, and branch-free payoff math. For `steps = 64`,
random number generation and math can dominate more strongly than payoff memory
traffic.

## Why Reduction Still Matters

Even when RNG dominates simulation time, the final price still requires reducing
all payoffs and payoff squares. CUB is used for those reductions so the
benchmark focuses on path simulation and honest end-to-end payoff aggregation
rather than a weak custom reduction bottleneck.

## Timing Scope

The main GPU latency metric includes:

```text
random draw/path simulation
payoff generation
payoff reduction
payoff-squared reduction
```

It excludes memory allocation, RNG state initialization, CUB temporary-storage
allocation, host-side result copies, and formatting. Setup time may be reported
separately later, but it is not mixed into the main latency number.

## Antithetic Variates

The antithetic variant uses each random draw `z` together with `-z`. Averaging
the two payoffs can reduce variance for the same requested path count, though
the actual standard-error change must be read from real benchmark output rather
than assumed.
