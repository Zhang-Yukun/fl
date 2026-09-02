#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_REPLAY_AFTER_EXP="${RUN_REPLAY_AFTER_EXP:-false}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/exp}"
REPLAY_INPUT_ROOT="${REPLAY_INPUT_ROOT:-${OUTPUT_PREFIX}}"
TASKS_RAW="${TASKS:-rare mnist cifar10}"
REPLAY_MODES_RAW="${REPLAY_MODES_RAW:-}"
SUITE_SEED="${SUITE_SEED:-42}"
BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"
REPLAY_RUN_DLG="${REPLAY_RUN_DLG:-true}"
REPLAY_RUN_IDLG="${REPLAY_RUN_IDLG:-true}"
REPLAY_FORCE="${REPLAY_FORCE:-false}"
REPLAY_OVERRIDE_RAW="${REPLAY_OVERRIDE_RAW:-}"

if [[ "${RUN_REPLAY_AFTER_EXP}" != "true" ]]; then
  exit 0
fi

if [[ "${REPLAY_RUN_DLG}" != "true" && "${REPLAY_RUN_IDLG}" != "true" ]]; then
  echo "[skip] replay-tail disabled because both REPLAY_RUN_DLG and REPLAY_RUN_IDLG are false"
  exit 0
fi

cmd=(
  bash scripts/run_replay_saved_update_batch.sh
  --input-root "${REPLAY_INPUT_ROOT}"
  --tasks "${TASKS_RAW}"
  --seeds "${SUITE_SEED}"
  --algorithms "${BASE_ALGOS}"
)

if [[ -n "${REPLAY_MODES_RAW}" ]]; then
  cmd+=(--modes "${REPLAY_MODES_RAW}")
fi
if [[ "${REPLAY_RUN_DLG}" == "true" && "${REPLAY_RUN_IDLG}" != "true" ]]; then
  cmd+=(--dlg-only)
elif [[ "${REPLAY_RUN_DLG}" != "true" && "${REPLAY_RUN_IDLG}" == "true" ]]; then
  cmd+=(--idlg-only)
fi
if [[ "${REPLAY_FORCE}" == "true" ]]; then
  cmd+=(--force)
fi
if [[ -n "${REPLAY_OVERRIDE_RAW}" ]]; then
  IFS=';' read -r -a override_list <<< "${REPLAY_OVERRIDE_RAW}"
  for override in "${override_list[@]}"; do
    [[ -z "${override}" ]] && continue
    cmd+=(--override "${override}")
  done
fi

echo "[replay-tail] input_root=${REPLAY_INPUT_ROOT} tasks=${TASKS_RAW} modes=${REPLAY_MODES_RAW:-all} seeds=${SUITE_SEED} algorithms=${BASE_ALGOS}"
"${cmd[@]}"
