#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_4script_manual_adam_ega_tune_$(date +%Y%m%d_%H%M%S)}"

# Tuned EGA wrapper.
# Shared overrides: training optimizer=adam, evaluation.mode=protocol,
# data.shuffle_train=true, model.dropout=0.1, federated algorithms=fedavg,topk,ega,
# plus tuned EGA codec settings matching the stronger compression run.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_mse" PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-v2-adam-ega-tune" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq5 TRACKING_TAG=oracle-attackfreq5 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=5 LOSS_NAME=mse LOSS_TAG=mse TRAIN_OPTIMIZER=adam EVAL_MODE=protocol SHUFFLE_TRAIN=true MODEL_DROPOUT=0.1 FEDERATED_ALGORITHMS=ega EGA_TRACKING_LABEL=ega-ed128-up-q127 EGA_ARTIFACT_PATH=artifacts/ega/ega_ed128_dm_ega_pc_q127.pt EGA_ENCODED_DIM=128 EGA_HIDDEN_DIM=1024 EGA_RESIDUAL_BLOCKS=2 EGA_QUANTIZATION_LEVEL=127 EGA_DOWNLOAD_METHOD=dense bash scripts/run_oracle_suite.sh --modes centralized,single_sync
