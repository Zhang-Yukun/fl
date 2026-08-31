#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-outputs/exp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/exp}"
TASKS_RAW="${TASKS:-rare mnist cifar10}"
TASK_LOSS_MAP="${TASK_LOSS_MAP:-rare=mse,mae;mnist=cross_entropy;cifar10=cross_entropy}"
MODE="${MODE:-single_sync}"
SEEDS_RAW="${SEEDS:-42 4096 2026 8192}"
ALGORITHMS_RAW="${ALGORITHMS:-centralized fedavg topk ega}"
INCLUDE_OLD="${INCLUDE_OLD:-false}"

usage() {
  cat <<'USAGE'
Usage:
  INPUT_ROOT=outputs/exp \
  OUTPUT_ROOT=outputs/analysis/exp \
  TASKS="rare mnist cifar10" \
  TASK_LOSS_MAP="rare=mse,mae;mnist=cross_entropy;cifar10=cross_entropy" \
  MODE=single_sync \
  SEEDS="42 4096 2026 8192" \
  ALGORITHMS="centralized fedavg topk ega" \
  bash scripts/run_analyze_experiment_suite_batch.sh

Optional flags:
  --input-root PATH
  --output-root PATH
  --tasks "rare mnist cifar10" | rare,mnist,cifar10
  --task-loss-map "rare=mse,mae;mnist=cross_entropy;cifar10=cross_entropy"
  --mode NAME
  --seeds "42 4096 2026 8192" | 42,4096,2026,8192
  --algorithms "centralized fedavg topk ega" | centralized,fedavg,topk,ega
  --include-old
USAGE
}

parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  printf '%s\n' "${raw}"
}

lookup_named_map_value() {
  local raw="$1"
  local key="$2"
  local pair
  IFS=';' read -r -a pairs <<< "${raw}"
  for pair in "${pairs[@]}"; do
    pair="${pair// /}"
    [[ -z "${pair}" ]] && continue
    if [[ "${pair}" != *=* ]]; then
      echo "Invalid map entry: ${pair}" >&2
      exit 1
    fi
    if [[ "${pair%%=*}" == "${key}" ]]; then
      printf '%s\n' "${pair#*=}"
      return 0
    fi
  done
  return 1
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
    --tasks)
      TASKS_RAW="$2"
      shift 2
      ;;
    --task-loss-map)
      TASK_LOSS_MAP="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --seeds)
      SEEDS_RAW="$2"
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

read -r -a TASK_LIST <<< "$(parse_list "${TASKS_RAW}")"
read -r -a SEED_LIST <<< "$(parse_list "${SEEDS_RAW}")"
read -r -a ALGORITHM_LIST <<< "$(parse_list "${ALGORITHMS_RAW}")"

if [[ ${#TASK_LIST[@]} -eq 0 || ${#SEED_LIST[@]} -eq 0 || ${#ALGORITHM_LIST[@]} -eq 0 ]]; then
  echo "Tasks, seeds, and algorithms must not be empty." >&2
  exit 1
fi

resolve_input_dir() {
  local task="$1"
  local seed="$2"
  local loss="$3"
  local candidate
  for candidate in \
    "${INPUT_ROOT}/${task}/${MODE}/${seed}/${loss}" \
    "${INPUT_ROOT}/${task}/${MODE}/${seed}/noattack_${loss}" \
    "${INPUT_ROOT}/${task}/${MODE}/${seed}/attack_${loss}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

task_losses() {
  local task="$1"
  local losses_raw
  losses_raw="$(lookup_named_map_value "${TASK_LOSS_MAP}" "${task}")" || {
    echo "Missing task loss mapping for task=${task}" >&2
    exit 1
  }
  parse_list "${losses_raw}"
}

run_single_seed() {
  local task="$1"
  local seed="$2"
  local loss="$3"
  local input_dir="${INPUT_ROOT}/${task}/${MODE}/${seed}/${loss}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "[skip] task=${task} seed=${seed} loss=${loss} mode=${MODE} missing=${input_dir}"
    return 0
  fi
  local output_dir="${OUTPUT_ROOT}/${task}/${MODE}/${seed}/${loss}"
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
  echo "[analyze] task=${task} seed=${seed} loss=${loss} mode=${MODE}"
  PYTHONPATH=. "${cmd[@]}"
}

run_multi_seed() {
  local task="$1"
  local loss="$2"
  local output_dir="${OUTPUT_ROOT}/${task}/${MODE}/multiseed/${loss}"
  local -a cmd=(
    "${PYTHON_BIN}" -m fedlab.tools.analyze_experiment_suite
  )
  local found_inputs=0
  local seed
  for seed in "${SEED_LIST[@]}"; do
    local input_dir="${INPUT_ROOT}/${task}/${MODE}/${seed}/${loss}"
    if [[ -d "${input_dir}" ]]; then
      cmd+=("${input_dir}")
      found_inputs=1
    fi
  done
  if [[ ${found_inputs} -eq 0 ]]; then
    echo "[skip] task=${task} multiseed loss=${loss} mode=${MODE} no-inputs-found"
    return 0
  fi
  cmd+=(
    --loss "${loss}"
    --output-dir "${output_dir}"
    --algorithms
  )
  cmd+=("${ALGORITHM_LIST[@]}")
  if [[ "${INCLUDE_OLD}" == "true" ]]; then
    cmd+=(--include-old)
  fi
  echo "[analyze] task=${task} multiseed loss=${loss} mode=${MODE} seeds=${SEED_LIST[*]}"
  PYTHONPATH=. "${cmd[@]}"
}

for task in "${TASK_LIST[@]}"; do
  read -r -a LOSS_LIST <<< "$(task_losses "${task}")"
  if [[ ${#LOSS_LIST[@]} -eq 0 ]]; then
    echo "Losses for task=${task} must not be empty." >&2
    exit 1
  fi

  for loss in "${LOSS_LIST[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
      run_single_seed "${task}" "${seed}" "${loss}"
    done
    run_multi_seed "${task}" "${loss}"
  done
done
