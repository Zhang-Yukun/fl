#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SESSION_NAME="${SESSION_NAME:-ega_adam_noattack_$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/ega_adam_noattack_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-ega-adam-noattack-explore-v1}"
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
GPU0_DEVICE="${GPU0_DEVICE:-cuda:0}"
GPU1_DEVICE="${GPU1_DEVICE:-cuda:1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUNDS="${ROUNDS:-500}"
PATIENCE="${PATIENCE:-50}"
TRAIN_LR="${TRAIN_LR:-0.001}"

mkdir -p "${BASE_OUTPUT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session ${SESSION_NAME} already exists" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n mse   "cd $(pwd) && PYTHON_BIN='${PYTHON_BIN}' BASE_OUTPUT='${BASE_OUTPUT}/adam_mse' PROJECT_NAME='${PROJECT_NAME}' GROUP_NAME='$(basename "${BASE_OUTPUT}")-adam-mse' GPU_ID='${GPU0_ID}' RUNTIME_DEVICE='${GPU0_DEVICE}' LOSS_NAME='mse' ROUNDS='${ROUNDS}' PATIENCE='${PATIENCE}' TRAIN_LR='${TRAIN_LR}' bash scripts/run_ega_noattack_adam_sweep_worker.sh | tee '${BASE_OUTPUT}/gpu0_mse.log'"

tmux new-window -t "${SESSION_NAME}" -n mae   "cd $(pwd) && PYTHON_BIN='${PYTHON_BIN}' BASE_OUTPUT='${BASE_OUTPUT}/adam_mae' PROJECT_NAME='${PROJECT_NAME}' GROUP_NAME='$(basename "${BASE_OUTPUT}")-adam-mae' GPU_ID='${GPU1_ID}' RUNTIME_DEVICE='${GPU1_DEVICE}' LOSS_NAME='mae' ROUNDS='${ROUNDS}' PATIENCE='${PATIENCE}' TRAIN_LR='${TRAIN_LR}' bash scripts/run_ega_noattack_adam_sweep_worker.sh | tee '${BASE_OUTPUT}/gpu1_mae.log'"

echo "SESSION_NAME=${SESSION_NAME}"
echo "BASE_OUTPUT=${BASE_OUTPUT}"
echo "PROJECT_NAME=${PROJECT_NAME}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
