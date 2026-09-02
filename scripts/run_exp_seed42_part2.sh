#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SUITE_SEED="${SUITE_SEED:-42}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
BASE_PORT="${BASE_PORT:-58100}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/exp}"
TASKS_RAW="${TASKS:-rare mnist cifar10}"
read -r -a TASKS <<< "${TASKS_RAW}"
MODE="multi_sync"
BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"
REPLAY_MODES_RAW="${REPLAY_MODES_RAW:-${MODE}}"

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
    BASE_ALGOS="${BASE_ALGOS}"       LOSS_NAME="${loss}"       MODE_SET="${MODE}"       TASK_SET="${task}"       TASK_IN_BASE_OUTPUT=true       SUITE_SEED="${SUITE_SEED}"       RUNTIME_DEVICE="${RUNTIME_DEVICE}"       BASE_PORT="${BASE_PORT}"       BASE_OUTPUT_ROOT="${OUTPUT_PREFIX}/${task}/${MODE}/${SUITE_SEED}"       PROJECT_NAME="fl-${task}-${MODE}"       bash scripts/run_controlled_suite.sh "$@"
  done
done

OUTPUT_PREFIX="${OUTPUT_PREFIX}" TASKS="${TASKS_RAW}" SUITE_SEED="${SUITE_SEED}" BASE_ALGOS="${BASE_ALGOS}" REPLAY_MODES_RAW="${REPLAY_MODES_RAW}" bash scripts/run_exp_replay_tail.sh
