#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_noattack_mse_adam_single_sync_$(date +%Y%m%d_%H%M%S)}"
COMMON_PROJECT_NAME="${PROJECT_NAME:-rare-earth-fl-oracle-noattack-single_sync-adam}"
SUITE_SEED="${SUITE_SEED:-2026}"
COMMON_BASE_OUTPUT="${BASE_OUTPUT_ROOT}/noattack_mse"

# Baselines plus centralized and default EGA.
BASE_OUTPUT="${COMMON_BASE_OUTPUT}" \
SUITE_SEED="${SUITE_SEED}" \
PROJECT_NAME="${COMMON_PROJECT_NAME}" \
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" \
RUN_TAG=oracle_noattack \
TRACKING_TAG=oracle-noattack \
ATTACK_ENABLED=false \
LOSS_NAME=mse \
LOSS_TAG=mse \
TRAIN_OPTIMIZER=adam \
EVAL_MODE=protocol \
SHUFFLE_TRAIN=true \
MODEL_DROPOUT=0.1 \
FEDERATED_ALGORITHMS=all \
bash scripts/run_oracle_suite.sh --modes centralized,single_sync