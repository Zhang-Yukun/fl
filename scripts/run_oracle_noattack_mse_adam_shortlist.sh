#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_noattack_mse_adam_shortlist_$(date +%Y%m%d_%H%M%S)}"
COMMON_PROJECT_NAME="${PROJECT_NAME:-rare-earth-fl-oracle-noattack-shortlist-adam}"
COMMON_BASE_OUTPUT="${BASE_OUTPUT_ROOT}/noattack_mse"

# Non-EGA baselines plus centralized.
BASE_OUTPUT="${COMMON_BASE_OUTPUT}" \
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
FEDERATED_ALGORITHMS=fedavg,topk,qsgd,randomk,sign,adaptive,qint8 \
bash scripts/run_oracle_suite.sh --modes centralized,single_sync

# Default EGA config. The suite will derive the run name from the active EGA parameters.
BASE_OUTPUT="${COMMON_BASE_OUTPUT}" \
PROJECT_NAME="${COMMON_PROJECT_NAME}" \
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" \
RUN_TAG=oracle_noattack \
TRACKING_TAG=oracle-noattack \
RUN_CENTRALIZED=false \
ATTACK_ENABLED=false \
LOSS_NAME=mse \
LOSS_TAG=mse \
TRAIN_OPTIMIZER=adam \
EVAL_MODE=protocol \
SHUFFLE_TRAIN=true \
MODEL_DROPOUT=0.1 \
FEDERATED_ALGORITHMS=ega \
EGA_PRETRAIN_DEVICE="${RUNTIME_DEVICE:-cuda:0}" \
bash scripts/run_oracle_suite.sh --modes single_sync
