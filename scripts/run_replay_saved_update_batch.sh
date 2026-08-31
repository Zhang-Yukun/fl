#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-outputs/exp}"
ALGORITHMS_RAW="${ALGORITHMS:-fedavg topk ega}"
TASKS_RAW="${TASKS:-}"
MODES_RAW="${MODES:-}"
SEEDS_RAW="${SEEDS:-}"
LOSSES_RAW="${LOSSES:-}"
RUN_DLG="${RUN_DLG:-true}"
RUN_IDLG="${RUN_IDLG:-true}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
EXTRA_OVERRIDES=()

usage() {
  cat <<'USAGE'
Usage:
  INPUT_ROOT=outputs/exp   ALGORITHMS="fedavg topk ega"   RUN_DLG=true   RUN_IDLG=true   bash scripts/run_replay_saved_update_batch.sh

Optional flags:
  --input-root PATH
  --algorithms "fedavg topk ega" | fedavg,topk,ega
  --tasks "rare mnist cifar10" | rare,mnist,cifar10
  --modes "single_sync multi_sync" | single_sync,multi_sync
  --seeds "42 4096 2026 8192" | 42,4096,2026,8192
  --losses "mse mae cross_entropy" | mse,mae,cross_entropy
  --dlg-only
  --idlg-only
  --force
  --override KEY=VALUE
USAGE
}

parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  printf '%s
' "${raw}"
}

list_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ -z "${item}" ]] && continue
    if [[ "${item}" == "${needle}" ]]; then
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
    --algorithms)
      ALGORITHMS_RAW="$2"
      shift 2
      ;;
    --tasks)
      TASKS_RAW="$2"
      shift 2
      ;;
    --modes)
      MODES_RAW="$2"
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
    --dlg-only)
      RUN_DLG=true
      RUN_IDLG=false
      shift
      ;;
    --idlg-only)
      RUN_DLG=false
      RUN_IDLG=true
      shift
      ;;
    --force)
      SKIP_EXISTING=false
      shift
      ;;
    --override)
      EXTRA_OVERRIDES+=("$2")
      shift 2
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

read -r -a ALGORITHM_LIST <<< "$(parse_list "${ALGORITHMS_RAW}")"
read -r -a TASK_LIST <<< "$(parse_list "${TASKS_RAW}")"
read -r -a MODE_LIST <<< "$(parse_list "${MODES_RAW}")"
read -r -a SEED_LIST <<< "$(parse_list "${SEEDS_RAW}")"
read -r -a LOSS_LIST <<< "$(parse_list "${LOSSES_RAW}")"

matches_filters() {
  local task="$1"
  local mode="$2"
  local seed="$3"
  local loss="$4"
  local algorithm="$5"
  if [[ ${#TASK_LIST[@]} -gt 0 ]] && ! list_contains "${task}" "${TASK_LIST[@]}"; then
    return 1
  fi
  if [[ ${#MODE_LIST[@]} -gt 0 ]] && ! list_contains "${mode}" "${MODE_LIST[@]}"; then
    return 1
  fi
  if [[ ${#SEED_LIST[@]} -gt 0 ]] && ! list_contains "${seed}" "${SEED_LIST[@]}"; then
    return 1
  fi
  if [[ ${#LOSS_LIST[@]} -gt 0 ]] && ! list_contains "${loss}" "${LOSS_LIST[@]}"; then
    return 1
  fi
  if [[ ${#ALGORITHM_LIST[@]} -gt 0 ]] && ! list_contains "${algorithm}" "${ALGORITHM_LIST[@]}"; then
    return 1
  fi
  return 0
}

discover_run_dirs() {
  find "${INPUT_ROOT}" -type f -path '*/saved_updates/index.json' | sort
}

run_method() {
  local method="$1"
  local run_dir="$2"
  local output_dir="${run_dir}/offline_attack_replay_${method}"
  if [[ "${SKIP_EXISTING}" == "true" && -f "${output_dir}/attack_summary.json" ]]; then
    echo "[skip] method=${method} run_dir=${run_dir} existing=${output_dir}/attack_summary.json"
    return 0
  fi
  local -a cmd=(
    "${PYTHON_BIN}" -m "fedlab.tools.replay_saved_update_${method}"
    "${run_dir}"
    --output-dir "${output_dir}"
  )
  local override
  for override in "${EXTRA_OVERRIDES[@]}"; do
    cmd+=(--override "${override}")
  done
  echo "[replay] method=${method} run_dir=${run_dir} output_dir=${output_dir}"
  PYTHONPATH=. "${cmd[@]}"
}

found_any=0
while IFS= read -r index_path; do
  [[ -z "${index_path}" ]] && continue
  found_any=1
  run_dir="$(dirname "$(dirname "${index_path}")")"
  rel_path="${run_dir#${INPUT_ROOT}/}"
  IFS='/' read -r -a parts <<< "${rel_path}"
  task="${parts[0]:-}"
  mode="${parts[1]:-}"
  seed="${parts[2]:-}"
  loss="${parts[3]:-}"
  algorithm="${parts[4]:-$(basename "${run_dir}")}"
  if ! matches_filters "${task}" "${mode}" "${seed}" "${loss}" "${algorithm}"; then
    echo "[skip] task=${task} mode=${mode} seed=${seed} loss=${loss} algorithm=${algorithm} filtered-out"
    continue
  fi
  if [[ "${RUN_DLG}" == "true" ]]; then
    run_method dlg "${run_dir}"
  fi
  if [[ "${RUN_IDLG}" == "true" ]]; then
    run_method idlg "${run_dir}"
  fi
done < <(discover_run_dirs)

if [[ ${found_any} -eq 0 ]]; then
  echo "[skip] no-saved-update-runs-found input_root=${INPUT_ROOT}"
fi
