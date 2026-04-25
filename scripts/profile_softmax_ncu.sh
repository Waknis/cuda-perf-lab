#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

cmake --preset configure-release
cmake --build --preset build-release

BIN="${ROOT_DIR}/build/release/softmax_bench"
OUT_DIR="${ROOT_DIR}/results/rtx_5060_ti/ncu"
mkdir -p "${OUT_DIR}"

ITERS="${ITERS:-5}"
WARMUP="${WARMUP:-2}"
SEED="${SEED:-1234}"

# Metric names can vary across Nsight Compute versions and GPU architectures.
# By default this uses Nsight's full section set. To force a metric list:
#
#   NCU_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second" ./scripts/profile_softmax_ncu.sh
#
base_args=(--target-processes all --force-overwrite)
if [[ -n "${NCU_METRICS:-}" ]]; then
  base_args+=(--metrics "${NCU_METRICS}")
else
  base_args+=(--set full)
fi

profiles=(
  "stable_two_pass 4096 512"
  "block_reduce 1024 4096"
  "warp_small_row 4096 128"
)

for profile in "${profiles[@]}"; do
  read -r variant rows cols <<< "${profile}"
  report="${OUT_DIR}/softmax_${variant}_rows${rows}_cols${cols}_iters${ITERS}"
  if ! ncu "${base_args[@]}" \
    --export "${report}" \
    "${BIN}" --variant "${variant}" --rows "${rows}" --cols "${cols}" --iters "${ITERS}" --warmup "${WARMUP}" --seed "${SEED}"; then
    if [[ -n "${NCU_METRICS:-}" ]]; then
      echo "ncu failed with NCU_METRICS='${NCU_METRICS}'. Retrying ${variant} with Nsight's default section set." >&2
      ncu --target-processes all --force-overwrite --set full \
        --export "${report}_set_full" \
        "${BIN}" --variant "${variant}" --rows "${rows}" --cols "${cols}" --iters "${ITERS}" --warmup "${WARMUP}" --seed "${SEED}"
    else
      echo "ncu failed for ${variant}. Some systems require administrator permissions or different metric sets." >&2
      exit 1
    fi
  fi
done

echo "Wrote Nsight Compute reports under ${OUT_DIR}"
