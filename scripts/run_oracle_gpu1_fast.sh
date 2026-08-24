#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_4script_manual_$(date +%Y%m%d_%H%M%S)}"

# Note: the underlying suite script sets CUDA_VISIBLE_DEVICES=${GPU_ID} internally.
# When selecting physical GPU 1, use runtime.device=cuda:0 inside that masked process.

# Fast lane: canonical attackfreq5 experiment, loss=mae.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_mae" PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-v1-fast" GPU_ID="${GPU_ID:-1}" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq5 TRACKING_TAG=oracle-attackfreq5 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=5 LOSS_NAME=mae LOSS_TAG=mae bash scripts/run_oracle_suite.sh --modes centralized,single_sync
