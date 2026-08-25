#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-outputs/output/exp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/exp}"
MODE="${MODE:-single_sync}"
PROFILE="${PROFILE:-noattack}"
SEEDS_RAW="${SEEDS:-42 55 2026 8192}"
LOSSES_RAW="${LOSSES:-mse mae}"
ALGORITHMS_RAW="${ALGORITHMS:-centralized fedavg topk ega}"
INCLUDE_OLD="${INCLUDE_OLD:-false}"

usage() {
  cat <<'USAGE'
Usage:
  INPUT_ROOT=outputs/output/exp \
  OUTPUT_ROOT=outputs/analysis/exp \
  MODE=single_sync \
  PROFILE=noattack \
  SEEDS="42 55 2026 8192" \
  LOSSES="mse mae" \
  ALGORITHMS="centralized fedavg topk ega" \
  bash scripts/run_analyze_experiment_suite_batch.sh

Optional flags:
  --input-root PATH
  --output-root PATH
  --mode NAME
  --profile noattack|attack
  --seeds "42 55 2026 8192" | 42,55,2026,8192
  --losses "mse mae" | mse,mae
  --algorithms "centralized fedavg topk ega" | centralized,fedavg,topk,ega
  --include-old
USAGE
}

parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  printf '%s\n' "${raw}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --seeds)
      SEEDS_RAW="$2"
      shift 2
      ;;
    --losses)
      LOSSES_RAW="$2"
      shift 2
      ;;
    --algorithms)
      ALGORITHMS_RAW="$2"
      shift 2
      ;;
    --include-old)
      INCLUDE_OLD=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${PROFILE}" in
  noattack|attack) ;;
  *)
    echo "Unsupported PROFILE=${PROFILE}. Use noattack or attack." >&2
    exit 1
    ;;
esac

read -r -a SEED_LIST <<< "$(parse_list "${SEEDS_RAW}")"
read -r -a LOSS_LIST <<< "$(parse_list "${LOSSES_RAW}")"
read -r -a ALGORITHM_LIST <<< "$(parse_list "${ALGORITHMS_RAW}")"

if [[ ${#SEED_LIST[@]} -eq 0 || ${#LOSS_LIST[@]} -eq 0 || ${#ALGORITHM_LIST[@]} -eq 0 ]]; then
  echo "Seeds, losses, and algorithms must not be empty." >&2
  exit 1
fi

run_single_seed() {
  local seed="$1"
  local loss="$2"
  local input_dir="${INPUT_ROOT}/${MODE}/${seed}/${PROFILE}_${loss}"
  local output_dir="${OUTPUT_ROOT}/${MODE}/${seed}/${PROFILE}_${loss}"
  local -a cmd=(
    "${PYTHON_BIN}" -m fedlab.tools.analyze_experiment_suite
    "${input_dir}"
    --loss "${loss}"
    --output-dir "${output_dir}"
    --algorithms
  )
  cmd+=("${ALGORITHM_LIST[@]}")
  if [[ "${INCLUDE_OLD}" == "true" ]]; then
    cmd+=(--include-old)
  fi
  echo "[analyze] seed=${seed} loss=${loss} profile=${PROFILE} mode=${MODE}"
  PYTHONPATH=. "${cmd[@]}"
}

run_multi_seed() {
  local loss="$1"
  local output_dir="${OUTPUT_ROOT}/${MODE}/multiseed/${PROFILE}_${loss}"
  local -a cmd=(
    "${PYTHON_BIN}" -m fedlab.tools.analyze_experiment_suite
  )
  local seed
  for seed in "${SEED_LIST[@]}"; do
    cmd+=("${INPUT_ROOT}/${MODE}/${seed}/${PROFILE}_${loss}")
  done
  cmd+=(
    --loss "${loss}"
    --output-dir "${output_dir}"
    --algorithms
  )
  cmd+=("${ALGORITHM_LIST[@]}")
  if [[ "${INCLUDE_OLD}" == "true" ]]; then
    cmd+=(--include-old)
  fi
  echo "[analyze] multiseed loss=${loss} profile=${PROFILE} mode=${MODE} seeds=${SEED_LIST[*]}"
  PYTHONPATH=. "${cmd[@]}"
}

for loss in "${LOSS_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    run_single_seed "${seed}" "${loss}"
  done
  run_multi_seed "${loss}"
done
