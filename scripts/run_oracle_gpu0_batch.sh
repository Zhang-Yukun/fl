#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_4script_manual_$(date +%Y%m%d_%H%M%S)}"

# Group A: no-attack baselines.
# Attack disabled, loss=mse.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/noattack" PROJECT_NAME="rare-earth-fl-oracle-noattack-v1" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_noattack TRACKING_TAG=oracle-noattack ATTACK_ENABLED=false LOSS_NAME=mse LOSS_TAG=mse bash scripts/run_oracle_suite.sh --modes centralized,single_sync

# Attack disabled, loss=mae.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/noattack_mae" PROJECT_NAME="rare-earth-fl-oracle-noattack-v1" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_noattack TRACKING_TAG=oracle-noattack ATTACK_ENABLED=false LOSS_NAME=mae LOSS_TAG=mae bash scripts/run_oracle_suite.sh --modes centralized,single_sync

# Group B: aggressive attack schedule.
# Attack enabled, frequency=1 round, loss=mse.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq1" PROJECT_NAME="rare-earth-fl-oracle-attackfreq1-v1" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq1 TRACKING_TAG=oracle-attackfreq1 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=1 LOSS_NAME=mse LOSS_TAG=mse bash scripts/run_oracle_suite.sh --modes centralized,single_sync

# Attack enabled, frequency=1 round, loss=mae.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq1_mae" PROJECT_NAME="rare-earth-fl-oracle-attackfreq1-v1" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq1 TRACKING_TAG=oracle-attackfreq1 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=1 LOSS_NAME=mae LOSS_TAG=mae bash scripts/run_oracle_suite.sh --modes centralized,single_sync
