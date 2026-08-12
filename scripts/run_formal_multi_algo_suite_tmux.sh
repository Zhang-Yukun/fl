#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SESSION_NAME="${SESSION_NAME:-formal_suite_multi}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/formal_multi_algo_suite_$(date +%Y%m%d_%H%M%S)}"
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
GPU0_DEVICE="${GPU0_DEVICE:-cuda:0}"
GPU1_DEVICE="${GPU1_DEVICE:-cuda:0}"
BASE_PORT0="${BASE_PORT0:-59000}"
BASE_PORT1="${BASE_PORT1:-59100}"
ROUNDS="${ROUNDS:-300}"
ATTACK_FREQUENCY="${ATTACK_FREQUENCY:-5}"
ATTACK_STEPS="${ATTACK_STEPS:-200}"
ATTACK_LR="${ATTACK_LR:-0.02}"
TRAIN_LR="${TRAIN_LR:-0.001}"

mkdir -p "${BASE_OUTPUT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session ${SESSION_NAME} already exists" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n gpu0_sync   "cd $(pwd) && BASE_OUTPUT='${BASE_OUTPUT}' GPU_ID='${GPU0_ID}' RUNTIME_DEVICE='${GPU0_DEVICE}' BASE_PORT='${BASE_PORT0}' WORKER_KIND=sync ROUNDS='${ROUNDS}' ATTACK_FREQUENCY='${ATTACK_FREQUENCY}' ATTACK_STEPS='${ATTACK_STEPS}' ATTACK_LR='${ATTACK_LR}' TRAIN_LR='${TRAIN_LR}' bash scripts/run_formal_multi_algo_worker.sh | tee '${BASE_OUTPUT}/gpu0_sync.log'"

tmux new-window -t "${SESSION_NAME}" -n gpu1_async   "cd $(pwd) && BASE_OUTPUT='${BASE_OUTPUT}' GPU_ID='${GPU1_ID}' RUNTIME_DEVICE='${GPU1_DEVICE}' BASE_PORT='${BASE_PORT1}' WORKER_KIND=async ROUNDS='${ROUNDS}' ATTACK_FREQUENCY='${ATTACK_FREQUENCY}' ATTACK_STEPS='${ATTACK_STEPS}' ATTACK_LR='${ATTACK_LR}' TRAIN_LR='${TRAIN_LR}' bash scripts/run_formal_multi_algo_worker.sh | tee '${BASE_OUTPUT}/gpu1_async.log'"

echo "SESSION_NAME=${SESSION_NAME}"
echo "BASE_OUTPUT=${BASE_OUTPUT}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
