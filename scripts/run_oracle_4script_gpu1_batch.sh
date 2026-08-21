#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_4script_manual_$(date +%Y%m%d_%H%M%S)}"

# Note: the underlying suite script sets CUDA_VISIBLE_DEVICES=${GPU_ID} internally.
# When selecting physical GPU 1, use runtime.device=cuda:0 inside that masked process.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_1000r_pat100" \
PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-1000r-pat100-v1" \
GPU_ID="${GPU_ID:-1}" \
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" \
bash scripts/run_oracle_attackfreq5_1000r_pat100_9algo_4mode_suite.sh

BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5" \
PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-v1" \
GPU_ID="${GPU_ID:-1}" \
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" \
bash scripts/run_oracle_attackfreq5_9algo_4mode_suite.sh

BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_maxsamples8" \
PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-maxsamples8-v1" \
GPU_ID="${GPU_ID:-1}" \
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" \
bash scripts/run_oracle_attackfreq5_maxsamples8_9algo_4mode_suite.sh
