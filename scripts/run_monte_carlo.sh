#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

cmake --preset configure-release
cmake --build --preset build-release

BIN="${ROOT_DIR}/build/release/monte_carlo_bench"
OUT_DIR="${ROOT_DIR}/results/rtx_5060_ti"
CSV_OUT="${OUT_DIR}/monte_carlo_results.csv"
JSON_OUT="${OUT_DIR}/monte_carlo_results.json"
ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-10}"
SEED="${SEED:-1234}"

mkdir -p "${OUT_DIR}"

configs=(
  "1048576 1"
  "1048576 64"
  "16777216 1"
  "16777216 64"
)

first=1
for config in "${configs[@]}"; do
  read -r paths steps <<< "${config}"
  if [[ "${first}" -eq 1 ]]; then
    "${BIN}" --variant all --paths "${paths}" --steps "${steps}" --iters "${ITERS}" --warmup "${WARMUP}" --seed "${SEED}" --csv > "${CSV_OUT}"
    first=0
  else
    "${BIN}" --variant all --paths "${paths}" --steps "${steps}" --iters "${ITERS}" --warmup "${WARMUP}" --seed "${SEED}" --csv | tail -n +2 >> "${CSV_OUT}"
  fi
done

json_tmp="$(mktemp)"
cleanup() {
  rm -f "${json_tmp}"
}
trap cleanup EXIT

printf '[\n' > "${JSON_OUT}"
first_json_row=1
for config in "${configs[@]}"; do
  read -r paths steps <<< "${config}"
  "${BIN}" --variant all --paths "${paths}" --steps "${steps}" --iters "${ITERS}" --warmup "${WARMUP}" --seed "${SEED}" --json > "${json_tmp}"
  mapfile -t json_rows < <(grep '^  {' "${json_tmp}" | sed 's/,$//')
  if [[ "${#json_rows[@]}" -eq 0 ]]; then
    echo "No JSON result rows produced for paths=${paths} steps=${steps}" >&2
    exit 1
  fi
  for row in "${json_rows[@]}"; do
    if [[ "${first_json_row}" -eq 0 ]]; then
      printf ',\n' >> "${JSON_OUT}"
    fi
    printf '%s' "${row}" >> "${JSON_OUT}"
    first_json_row=0
  done
done
printf '\n]\n' >> "${JSON_OUT}"

echo "Wrote ${CSV_OUT}"
echo "Wrote ${JSON_OUT}"
