#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE_SEED="${SUITE_SEED:-42}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
BASE_PORT="${BASE_PORT:-58000}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/exp}"
TASKS=(rare mnist cifar10)
PROFILE="noattack"
MODES=(single_sync multi_sync)

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
    for mode in "${MODES[@]}"; do
      mode_base_port="${BASE_PORT}"
      if [[ "${mode}" == "multi_sync" ]]; then
        mode_base_port="$((BASE_PORT + 100))"
      fi
      PROFILE="${PROFILE}"       BASE_ALGOS=fedavg,topk,ega       LOSS_NAME="${loss}"       MODE_SET="${mode}"       TASK_SET="${task}"       TASK_IN_BASE_OUTPUT=true       SUITE_SEED="${SUITE_SEED}"       RUNTIME_DEVICE="${RUNTIME_DEVICE}"       BASE_PORT="${mode_base_port}"       BASE_OUTPUT_ROOT="${OUTPUT_PREFIX}/${task}/${mode}/${SUITE_SEED}"       PROJECT_NAME="fl-${task}-${mode}-${PROFILE}"       bash scripts/run_controlled_suite.sh
    done
  done
done
