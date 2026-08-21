#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SESSION_NAME="${SESSION_NAME:-ega_target_$(date +%Y%m%d_%H%M%S)}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/ega_target_$(date +%Y%m%d_%H%M%S)}"
GPU0_ID="${GPU0_ID:-0}"
GPU1_ID="${GPU1_ID:-1}"
GPU0_DEVICE="${GPU0_DEVICE:-cuda:0}"
GPU1_DEVICE="${GPU1_DEVICE:-cuda:1}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-outputs/suite500_20260820_161030/fedavg_single_sync_500r_pat50/summary.json}"

mkdir -p "${BASE_OUTPUT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session ${SESSION_NAME} already exists" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" -n gpu0 \
  "cd $(pwd) && BASE_OUTPUT='${BASE_OUTPUT}' GPU_ID='${GPU0_ID}' RUNTIME_DEVICE='${GPU0_DEVICE}' WORKER_KIND='gpu0' BASELINE_SUMMARY='${BASELINE_SUMMARY}' bash scripts/run_ega_target_sweep_worker.sh | tee '${BASE_OUTPUT}/gpu0.log'"

tmux new-window -t "${SESSION_NAME}" -n gpu1 \
  "cd $(pwd) && BASE_OUTPUT='${BASE_OUTPUT}' GPU_ID='${GPU1_ID}' RUNTIME_DEVICE='${GPU1_DEVICE}' WORKER_KIND='gpu1' BASELINE_SUMMARY='${BASELINE_SUMMARY}' bash scripts/run_ega_target_sweep_worker.sh | tee '${BASE_OUTPUT}/gpu1.log'"

echo "SESSION_NAME=${SESSION_NAME}"
echo "BASE_OUTPUT=${BASE_OUTPUT}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
