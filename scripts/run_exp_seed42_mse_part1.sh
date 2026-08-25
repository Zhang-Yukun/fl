#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE_SEED="${SUITE_SEED:-42}"
LOSS_NAME="${LOSS_NAME:-mse}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-adam}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/exp}"
ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS:-5}"

MODES=(single_sync single_async)

for mode in "${MODES[@]}"; do
  PROFILE=noattack   BASE_ALGOS=fedavg,topk,ega   LOSS_NAME="${LOSS_NAME}"   MODE_SET="${mode}"   SUITE_SEED="${SUITE_SEED}"   RUNTIME_DEVICE="${RUNTIME_DEVICE}"   TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER}"   BASE_OUTPUT_ROOT="${OUTPUT_PREFIX}/${mode}/${SUITE_SEED}"   PROJECT_NAME="re_fl_noattack_${mode}_adam"   bash scripts/run_controlled_suite.sh

  PROFILE=attack   LOSS_NAME="${LOSS_NAME}"   MODE_SET="${mode}"   SUITE_SEED="${SUITE_SEED}"   RUNTIME_DEVICE="${RUNTIME_DEVICE}"   TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER}"   ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS}"   BASE_OUTPUT_ROOT="${OUTPUT_PREFIX}/${mode}/${SUITE_SEED}"   PROJECT_NAME="re_fl_attack_${mode}_adam"   bash scripts/run_controlled_suite.sh
done
