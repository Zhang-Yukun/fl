#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_4script_manual_$(date +%Y%m%d_%H%M%S)}"

BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq1" PROJECT_NAME="rare-earth-fl-oracle-attackfreq1-mse-v1" GPU_ID="${GPU_ID:-0}" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" bash scripts/run_oracle_attackfreq1_9algo_4mode_suite.sh

BASE_OUTPUT="${BASE_OUTPUT_ROOT}/noattack" PROJECT_NAME="rare-earth-fl-oracle-noattack-mse-v1" GPU_ID="${GPU_ID:-0}" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" bash scripts/run_oracle_noattack_9algo_4mode_suite.sh

BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq1_mae" PROJECT_NAME="rare-earth-fl-oracle-attackfreq1-mae-v1" GPU_ID="${GPU_ID:-0}" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" bash scripts/run_oracle_attackfreq1_9algo_4mode_suite_mae.sh

BASE_OUTPUT="${BASE_OUTPUT_ROOT}/noattack_mae" PROJECT_NAME="rare-earth-fl-oracle-noattack-mae-v1" GPU_ID="${GPU_ID:-0}" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" bash scripts/run_oracle_noattack_9algo_4mode_suite_mae.sh
