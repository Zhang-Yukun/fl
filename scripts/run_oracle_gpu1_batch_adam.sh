#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_4script_manual_adam_$(date +%Y%m%d_%H%M%S)}"

# Oracle suite wrapper with the old strong EGA-style protocol settings.
# Shared overrides: training optimizer=adam, loss=mse, evaluation.mode=protocol,
# data.shuffle_train=true, model.dropout=0.1, federated algorithms=fedavg,topk,ega.

BASE_OUTPUT="${BASE_OUTPUT_ROOT}/noattack_mse" PROJECT_NAME="rare-earth-fl-oracle-noattack-v2-adam" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_noattack TRACKING_TAG=oracle-noattack ATTACK_ENABLED=false LOSS_NAME=mse LOSS_TAG=mse TRAIN_OPTIMIZER=adam EVAL_MODE=protocol SHUFFLE_TRAIN=true MODEL_DROPOUT=0.1 FEDERATED_ALGORITHMS=all bash scripts/run_oracle_suite.sh --modes centralized,single_sync

BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_mse" PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-v2-adam" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq5 TRACKING_TAG=oracle-attackfreq5 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=5 LOSS_NAME=mse LOSS_TAG=mse TRAIN_OPTIMIZER=adam EVAL_MODE=protocol SHUFFLE_TRAIN=true MODEL_DROPOUT=0.1 FEDERATED_ALGORITHMS=fedavg,topk,ega bash scripts/run_oracle_suite.sh --modes centralized,single_sync


