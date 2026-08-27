#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-noattack}"
LOSS_NAME="${LOSS_NAME:-mse}"
LOSS_TAG="${LOSS_TAG:-${LOSS_NAME}}"
MODE_SET="${MODE_SET:-all}"
TASK_SET="${TASK_SET:-rare}"
TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS:-rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10}"
TASK_CLIENT_IDS="${TASK_CLIENT_IDS:-rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3}"
TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS:-rare}"
SUITE_SEED="${SUITE_SEED:-2026}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
RUN_CENTRALIZED="${RUN_CENTRALIZED:-true}"
ROUNDS="${ROUNDS:-}"
PATIENCE="${PATIENCE:-}"
BASE_PORT="${BASE_PORT:-58000}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"
ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS:-10}"
TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-adam}"
EGA_ARTIFACT_PATH="${EGA_ARTIFACT_PATH:-}"
EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE:-}"
EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS:-}"
SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-true}"
MODEL_DROPOUT="${MODEL_DROPOUT:-0.1}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/suite_${PROFILE}_${LOSS_NAME}_seed${SUITE_SEED}_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-fl-suite-${PROFILE}-${LOSS_NAME}}"

usage() {
  cat <<'USAGE'
Usage:
  PROFILE=noattack|attack   LOSS_NAME=mse|mae   TASK_SET=task1,task2|all   TASK_CONFIG_DIRS="rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10"   TASK_CLIENT_IDS="rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3"   TASK_LOSS_OVERRIDE_TASKS=rare   MODE_SET=all|single_sync|single_async|multi_sync|multi_async|centralized|comma,list   SUITE_SEED=2026   RUNTIME_DEVICE=cuda:0   ROUNDS=10   PATIENCE=500   BASE_PORT=58000   STARTUP_WAIT_SECONDS=60   EGA_ARTIFACT_PATH=artifacts/ega/ega_h240_v1.pt   EGA_PRETRAIN_DEVICE=same   EGA_PRETRAIN_EPOCHS=100   BASE_OUTPUT_ROOT=outputs/my_suite   PROJECT_NAME=my-wandb-project   bash scripts/run_controlled_suite.sh

Notes:
  - PROFILE=noattack runs centralized + fedavg/topk/ega for the selected tasks.
  - PROFILE=attack runs centralized + fedavg/topk/ega with attack enabled for the selected tasks.
  - TASK_SET uses keys from TASK_CONFIG_DIRS. all expands to every mapped task.
  - MODE_SET uses user-facing names. multi_sync -> grpc_sync, multi_async -> grpc_async.
  - Set RUN_CENTRALIZED=false if you want to skip centralized.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

case "${PROFILE}" in
  noattack|attack) ;;
  *)
    echo "Unsupported PROFILE=${PROFILE}. Use noattack or attack." >&2
    exit 1
    ;;
esac

case "${LOSS_NAME}" in
  mse|mae) ;;
  *)
    echo "Unsupported LOSS_NAME=${LOSS_NAME}. Use mse or mae." >&2
    exit 1
    ;;
esac

map_modes() {
  local raw="${1:-all}"
  local include_single_sync=false
  local include_single_async=false
  local include_grpc_sync=false
  local include_grpc_async=false
  IFS=',' read -r -a parts <<< "${raw}"
  local item
  for item in "${parts[@]}"; do
    item="${item// /}"
    [[ -z "${item}" ]] && continue
    case "${item}" in
      all)
        include_single_sync=true
        include_single_async=true
        include_grpc_sync=true
        include_grpc_async=true
        ;;
      single_sync)
        include_single_sync=true
        ;;
      single_async)
        include_single_async=true
        ;;
      multi_sync|grpc_sync)
        include_grpc_sync=true
        ;;
      multi_async|grpc_async)
        include_grpc_async=true
        ;;
      centralized)
        ;;
      none)
        ;;
      *)
        echo "Unsupported MODE_SET entry: ${item}" >&2
        exit 1
        ;;
    esac
  done

  local modes=()
  [[ "${include_single_sync}" == true ]] && modes+=(single_sync)
  [[ "${include_single_async}" == true ]] && modes+=(single_async)
  [[ "${include_grpc_sync}" == true ]] && modes+=(grpc_sync)
  [[ "${include_grpc_async}" == true ]] && modes+=(grpc_async)

  if [[ ${#modes[@]} -eq 0 ]]; then
    printf 'none\n'
    return
  fi

  local joined="${modes[0]}"
  local idx
  for ((idx=1; idx<${#modes[@]}; idx++)); do
    joined+=",${modes[idx]}"
  done
  printf '%s\n' "${joined}"
}

suite_modes() {
  local federated_modes="$1"
  if [[ "${RUN_CENTRALIZED}" != "true" ]]; then
    printf '%s\n' "${federated_modes}"
    return
  fi

  if [[ "${federated_modes}" == "none" ]]; then
    printf 'centralized\n'
    return
  fi

  printf 'centralized,%s\n' "${federated_modes}"
}

MAPPED_MODES="$(map_modes "${MODE_SET}")"
SUITE_MODES="$(suite_modes "${MAPPED_MODES}")"

if [[ "${PROFILE}" == "noattack" ]]; then
  RUN_TAG="${RUN_TAG:-oracle_noattack}"
  TRACKING_TAG="${TRACKING_TAG:-oracle-noattack}"
  ATTACK_ENABLED=false
  BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"
else
  RUN_TAG="${RUN_TAG:-oracle_attackfreq${ATTACK_FREQUENCY_ROUNDS}}"
  TRACKING_TAG="${TRACKING_TAG:-oracle-attackfreq${ATTACK_FREQUENCY_ROUNDS}}"
  ATTACK_ENABLED=true
  BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"
fi

COMMON_BASE_OUTPUT="${BASE_OUTPUT_ROOT}/${PROFILE}_${LOSS_NAME}"

env \
  BASE_OUTPUT="${COMMON_BASE_OUTPUT}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  SUITE_SEED="${SUITE_SEED}" \
  RUNTIME_DEVICE="${RUNTIME_DEVICE}" \
  ROUNDS="${ROUNDS}" \
  PATIENCE="${PATIENCE}" \
  BASE_PORT="${BASE_PORT}" \
  STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS}" \
  RUN_TAG="${RUN_TAG}" \
  TRACKING_TAG="${TRACKING_TAG}" \
  RUN_CENTRALIZED="${RUN_CENTRALIZED}" \
  ATTACK_ENABLED="${ATTACK_ENABLED}" \
  ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS}" \
  LOSS_NAME="${LOSS_NAME}" \
  LOSS_TAG="${LOSS_TAG}" \
  TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER}" \
  EGA_ARTIFACT_PATH="${EGA_ARTIFACT_PATH}" \
  EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE}" \
  EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS}" \
  SHUFFLE_TRAIN="${SHUFFLE_TRAIN}" \
  MODEL_DROPOUT="${MODEL_DROPOUT}" \
  TASK_SET="${TASK_SET}" \
  TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS}" \
  TASK_CLIENT_IDS="${TASK_CLIENT_IDS}" \
  TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS}" \
  FEDERATED_ALGORITHMS="${BASE_ALGOS}" \
  bash scripts/run_suite.sh --modes "${SUITE_MODES}" --tasks "${TASK_SET}"
