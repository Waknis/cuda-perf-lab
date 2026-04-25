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
