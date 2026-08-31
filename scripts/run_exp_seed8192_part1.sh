#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE_SEED="${SUITE_SEED:-8192}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
BASE_PORT="${BASE_PORT:-61000}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/exp}"
TASKS_RAW="${TASKS:-rare mnist cifar10}"
read -r -a TASKS <<< "${TASKS_RAW}"
PROFILE="noattack"
MODE="single_sync"

for task in "${TASKS[@]}"; do
  case "${task}" in
    rare)
      LOSSES=(mse)
      ;;
    *)
      LOSSES=(cross_entropy)
      ;;
  esac

  for loss in "${LOSSES[@]}"; do
    PROFILE="${PROFILE}"       BASE_ALGOS=fedavg,topk,ega       LOSS_NAME="${loss}"       MODE_SET="${MODE}"       TASK_SET="${task}"       TASK_IN_BASE_OUTPUT=true       SUITE_SEED="${SUITE_SEED}"       RUNTIME_DEVICE="${RUNTIME_DEVICE}"       BASE_PORT="${BASE_PORT}"       BASE_OUTPUT_ROOT="${OUTPUT_PREFIX}/${task}/${MODE}/${SUITE_SEED}"       PROJECT_NAME="fl-${task}-${MODE}-${PROFILE}"       bash scripts/run_controlled_suite.sh
  done
done
