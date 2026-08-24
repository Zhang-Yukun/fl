#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/oracle_4script_manual_$(date +%Y%m%d_%H%M%S)}"

# Group C: attack frequency fixed at 5 rounds, compare sample budget.
# Attack enabled, frequency=5 rounds, max_samples=8, loss=mse.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_maxsamples8" PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-maxsamples8-v1" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq5_maxsamples8 TRACKING_TAG=oracle-attackfreq5-maxsamples8 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=5 ATTACK_MAX_SAMPLES=8 LOSS_NAME=mse LOSS_TAG=mse bash scripts/run_oracle_suite.sh --modes centralized,single_sync

# Attack enabled, frequency=5 rounds, max_samples=8, loss=mae.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_maxsamples8_mae" PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-maxsamples8-v1" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq5_maxsamples8 TRACKING_TAG=oracle-attackfreq5-maxsamples8 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=5 ATTACK_MAX_SAMPLES=8 LOSS_NAME=mae LOSS_TAG=mae bash scripts/run_oracle_suite.sh --modes centralized,single_sync

# Group D: longer optimization horizon.
# Attack enabled, frequency=5 rounds, rounds=1000, patience=100, train_lr=0.0005, attack_lr=0.001, attack_optimizer=adam, loss=mse.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_1000r_pat100" PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-1000r-pat100-v1" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq5_1000r_pat100 TRACKING_TAG=oracle-attackfreq5-1000r-pat100 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=5 ROUNDS=1000 PATIENCE=100 TRAIN_LR=0.0005 ATTACK_LR=0.001 ATTACK_OPTIMIZER=adam LOSS_NAME=mse LOSS_TAG=mse bash scripts/run_oracle_suite.sh --modes centralized,single_sync

# Attack enabled, frequency=5 rounds, rounds=1000, patience=100, train_lr=0.0005, attack_lr=0.001, attack_optimizer=adam, loss=mae.
BASE_OUTPUT="${BASE_OUTPUT_ROOT}/attackfreq5_1000r_pat100_mae" PROJECT_NAME="rare-earth-fl-oracle-attackfreq5-1000r-pat100-v1" RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}" RUN_TAG=oracle_attackfreq5_1000r_pat100 TRACKING_TAG=oracle-attackfreq5-1000r-pat100 ATTACK_ENABLED=true ATTACK_FREQUENCY_ROUNDS=5 ROUNDS=1000 PATIENCE=100 TRAIN_LR=0.0005 ATTACK_LR=0.001 ATTACK_OPTIMIZER=adam LOSS_NAME=mae LOSS_TAG=mae bash scripts/run_oracle_suite.sh --modes centralized,single_sync
