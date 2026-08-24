#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SESSION_NAME="${SESSION_NAME:-ega_targeted_noattack_$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/ega_targeted_noattack_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME_MSE="${PROJECT_NAME_MSE:-rare-earth-ega-targeted-noattack-mse-v1}"
PROJECT_NAME_MAE="${PROJECT_NAME_MAE:-rare-earth-ega-targeted-noattack-mae-v1}"
GPU0_DEVICE="${GPU0_DEVICE:-cuda:0}"
GPU1_DEVICE="${GPU1_DEVICE:-cuda:0}"
GPU0_VISIBLE="${GPU0_VISIBLE:-0}"
GPU1_VISIBLE="${GPU1_VISIBLE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROUNDS="${ROUNDS:-300}"
PATIENCE="${PATIENCE:-50}"
TRAIN_LR="${TRAIN_LR:-0.001}"

mkdir -p "${BASE_OUTPUT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session ${SESSION_NAME} already exists" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n mse \
  "cd $(pwd) && CUDA_VISIBLE_DEVICES='${GPU0_VISIBLE}' PYTHON_BIN='${PYTHON_BIN}' BASE_OUTPUT='${BASE_OUTPUT}/mse' PROJECT_NAME='${PROJECT_NAME_MSE}' GROUP_NAME='$(basename "${BASE_OUTPUT}")-mse' RUNTIME_DEVICE='${GPU0_DEVICE}' LOSS_NAME='mse' ROUNDS='${ROUNDS}' PATIENCE='${PATIENCE}' TRAIN_LR='${TRAIN_LR}' bash scripts/run_ega_targeted_noattack_worker.sh | tee '${BASE_OUTPUT}/gpu0_mse.log'"

tmux new-window -t "${SESSION_NAME}" -n mae \
  "cd $(pwd) && CUDA_VISIBLE_DEVICES='${GPU1_VISIBLE}' PYTHON_BIN='${PYTHON_BIN}' BASE_OUTPUT='${BASE_OUTPUT}/mae' PROJECT_NAME='${PROJECT_NAME_MAE}' GROUP_NAME='$(basename "${BASE_OUTPUT}")-mae' RUNTIME_DEVICE='${GPU1_DEVICE}' LOSS_NAME='mae' ROUNDS='${ROUNDS}' PATIENCE='${PATIENCE}' TRAIN_LR='${TRAIN_LR}' bash scripts/run_ega_targeted_noattack_worker.sh | tee '${BASE_OUTPUT}/gpu1_mae.log'"

echo "SESSION_NAME=${SESSION_NAME}"
echo "BASE_OUTPUT=${BASE_OUTPUT}"
echo "PROJECT_NAME_MSE=${PROJECT_NAME_MSE}"
echo "PROJECT_NAME_MAE=${PROJECT_NAME_MAE}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
