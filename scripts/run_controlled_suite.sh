#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOSS_NAME="${LOSS_NAME:-mse}"
LOSS_TAG="${LOSS_TAG:-${LOSS_NAME}}"
MODE_SET="${MODE_SET:-all}"
TASK_SET="${TASK_SET:-rare}"
TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS:-rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10}"
TASK_CLIENT_IDS="${TASK_CLIENT_IDS:-rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3}"
TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS:-rare}"
TASK_IN_BASE_OUTPUT="${TASK_IN_BASE_OUTPUT:-false}"
SUITE_SEED="${SUITE_SEED:-2026}"
RUNTIME_SEED="${RUNTIME_SEED:-${SUITE_SEED}}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
RUN_CENTRALIZED="${RUN_CENTRALIZED:-true}"
ROUNDS="${ROUNDS:-}"
PATIENCE="${PATIENCE:-}"
BASE_PORT="${BASE_PORT:-58000}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"
CAPTURE_FREQUENCY_ROUNDS="${CAPTURE_FREQUENCY_ROUNDS:-}"
TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-}"
QSGD_SEED="${QSGD_SEED:-${SUITE_SEED}}"
RANDOMK_SEED="${RANDOMK_SEED:-${SUITE_SEED}}"
ADAPTIVE_RDP_SEED="${ADAPTIVE_RDP_SEED:-${SUITE_SEED}}"
QINT8_SEED="${QINT8_SEED:-${SUITE_SEED}}"
EGA_QUANTIZATION_SEED="${EGA_QUANTIZATION_SEED:-${SUITE_SEED}}"
EGA_ARTIFACT_PATH="${EGA_ARTIFACT_PATH:-}"
EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE:-}"
EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS:-}"
EGA_PRETRAIN_SEED="${EGA_PRETRAIN_SEED:-${SUITE_SEED}}"
SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-}"
MODEL_DROPOUT="${MODEL_DROPOUT:-}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/suite_${LOSS_NAME}_seed${SUITE_SEED}_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-fl-suite-${LOSS_NAME}}"

usage() {
  cat <<'USAGE'
Usage:
  LOSS_NAME=mse|mae|cross_entropy   TASK_SET=task1,task2|all   TASK_CONFIG_DIRS="rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10"   TASK_CLIENT_IDS="rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3"   TASK_LOSS_OVERRIDE_TASKS=rare   TASK_IN_BASE_OUTPUT=true   MODE_SET=all|single_sync|single_async|multi_sync|multi_async|centralized|comma,list   SUITE_SEED=2026   RUNTIME_SEED=7   QSGD_SEED=7   RANDOMK_SEED=7   ADAPTIVE_RDP_SEED=7   QINT8_SEED=7   EGA_QUANTIZATION_SEED=7   EGA_PRETRAIN_SEED=7   RUNTIME_DEVICE=cuda:0   ROUNDS=10   PATIENCE=500   BASE_PORT=58000   STARTUP_WAIT_SECONDS=60   CAPTURE_FREQUENCY_ROUNDS=30   EGA_ARTIFACT_PATH=artifacts/ega/ega_h240_v1.pt   EGA_PRETRAIN_DEVICE=same   EGA_PRETRAIN_EPOCHS=220   BASE_OUTPUT_ROOT=outputs/my_suite   PROJECT_NAME=my-wandb-project   bash scripts/run_controlled_suite.sh

Notes:
  - 训练脚本只负责 centralized + fedavg/topk/ega 训练与离线 replay 所需更新采集。
  - 攻击执行由独立 replay 脚本完成，不再区分 noattack/attack profile。
  - TASK_SET uses keys from TASK_CONFIG_DIRS. all expands to every mapped task.
  - MODE_SET uses user-facing names. multi_sync -> grpc_sync, multi_async -> grpc_async.
  - Set RUN_CENTRALIZED=false if you want to skip centralized.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

PASSTHROUGH_ARGS=("$@")

case "${LOSS_NAME}" in
  mse|mae|cross_entropy) ;;
  *)
    echo "Unsupported LOSS_NAME=${LOSS_NAME}. Use mse, mae, or cross_entropy." >&2
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

if [[ -n "${CAPTURE_FREQUENCY_ROUNDS}" ]]; then
  RUN_TAG="${RUN_TAG:-capturefreq${CAPTURE_FREQUENCY_ROUNDS}}"
  TRACKING_TAG="${TRACKING_TAG:-capturefreq${CAPTURE_FREQUENCY_ROUNDS}}"
else
  RUN_TAG="${RUN_TAG:-suite}"
  TRACKING_TAG="${TRACKING_TAG:-suite}"
fi
BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"

COMMON_BASE_OUTPUT="${BASE_OUTPUT_ROOT}/${LOSS_NAME}"

env \
  BASE_OUTPUT="${COMMON_BASE_OUTPUT}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  SUITE_SEED="${SUITE_SEED}" \
  RUNTIME_SEED="${RUNTIME_SEED}" \
  RUNTIME_DEVICE="${RUNTIME_DEVICE}" \
  ROUNDS="${ROUNDS}" \
  PATIENCE="${PATIENCE}" \
  BASE_PORT="${BASE_PORT}" \
  STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS}" \
  RUN_TAG="${RUN_TAG}" \
  TRACKING_TAG="${TRACKING_TAG}" \
  RUN_CENTRALIZED="${RUN_CENTRALIZED}" \
  CAPTURE_FREQUENCY_ROUNDS="${CAPTURE_FREQUENCY_ROUNDS}" \
  LOSS_NAME="${LOSS_NAME}" \
  LOSS_TAG="${LOSS_TAG}" \
  TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER}" \
  QSGD_SEED="${QSGD_SEED}" \
  RANDOMK_SEED="${RANDOMK_SEED}" \
  ADAPTIVE_RDP_SEED="${ADAPTIVE_RDP_SEED}" \
  QINT8_SEED="${QINT8_SEED}" \
  EGA_QUANTIZATION_SEED="${EGA_QUANTIZATION_SEED}" \
  EGA_ARTIFACT_PATH="${EGA_ARTIFACT_PATH}" \
  EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE}" \
  EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS}" \
  EGA_PRETRAIN_SEED="${EGA_PRETRAIN_SEED}" \
  SHUFFLE_TRAIN="${SHUFFLE_TRAIN}" \
  MODEL_DROPOUT="${MODEL_DROPOUT}" \
  TASK_SET="${TASK_SET}" \
  TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS}" \
  TASK_CLIENT_IDS="${TASK_CLIENT_IDS}" \
  TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS}" \
  TASK_IN_BASE_OUTPUT="${TASK_IN_BASE_OUTPUT}" \
  FEDERATED_ALGORITHMS="${BASE_ALGOS}" \
  bash scripts/run_suite.sh --modes "${SUITE_MODES}" --tasks "${TASK_SET}" "${PASSTHROUGH_ARGS[@]}"
