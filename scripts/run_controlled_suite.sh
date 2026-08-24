#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-noattack}"
LOSS_NAME="${LOSS_NAME:-mse}"
LOSS_TAG="${LOSS_TAG:-${LOSS_NAME}}"
MODE_SET="${MODE_SET:-all}"
SUITE_SEED="${SUITE_SEED:-2026}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
RUN_CENTRALIZED="${RUN_CENTRALIZED:-true}"
ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS:-5}"
TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-adam}"
EVAL_MODE="${EVAL_MODE:-protocol}"
SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-true}"
MODEL_DROPOUT="${MODEL_DROPOUT:-0.1}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/suite_${PROFILE}_${LOSS_NAME}_seed${SUITE_SEED}_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-fl-suite-${PROFILE}-${LOSS_NAME}}"

usage() {
  cat <<'USAGE'
Usage:
  PROFILE=noattack|attack   LOSS_NAME=mse|mae   MODE_SET=all|single_sync|single_async|multi_sync|multi_async|centralized|comma,list   SUITE_SEED=2026   RUNTIME_DEVICE=cuda:0   BASE_OUTPUT_ROOT=outputs/my_suite   PROJECT_NAME=my-wandb-project   bash scripts/run_controlled_suite.sh

Notes:
  - PROFILE=noattack runs centralized + all federated algorithms in the base suite,
    then appends the shortlisted EGA configs for the selected loss.
  - PROFILE=attack runs centralized + fedavg/topk/ega in the base suite,
    then appends the shortlisted EGA configs for the selected loss.
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

HAS_FEDERATED_MODES=true
MAPPED_MODES="$(map_modes "${MODE_SET}")"
if [[ "${MAPPED_MODES}" == "none" ]]; then
  HAS_FEDERATED_MODES=false
fi
SUITE_MODES="$(suite_modes "${MAPPED_MODES}")"

if [[ "${PROFILE}" == "noattack" ]]; then
  RUN_TAG="${RUN_TAG:-oracle_noattack}"
  TRACKING_TAG="${TRACKING_TAG:-oracle-noattack}"
  ATTACK_ENABLED=false
  BASE_ALGOS="${BASE_ALGOS:-all}"
else
  RUN_TAG="${RUN_TAG:-oracle_attackfreq${ATTACK_FREQUENCY_ROUNDS}}"
  TRACKING_TAG="${TRACKING_TAG:-oracle-attackfreq${ATTACK_FREQUENCY_ROUNDS}}"
  ATTACK_ENABLED=true
  BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"
fi

COMMON_BASE_OUTPUT="${BASE_OUTPUT_ROOT}/${PROFILE}_${LOSS_NAME}"

run_base_suite() {
  BASE_OUTPUT="${COMMON_BASE_OUTPUT}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  SUITE_SEED="${SUITE_SEED}" \
  RUNTIME_DEVICE="${RUNTIME_DEVICE}" \
  RUN_TAG="${RUN_TAG}" \
  TRACKING_TAG="${TRACKING_TAG}" \
  RUN_CENTRALIZED="${RUN_CENTRALIZED}" \
  ATTACK_ENABLED="${ATTACK_ENABLED}" \
  ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS}" \
  LOSS_NAME="${LOSS_NAME}" \
  LOSS_TAG="${LOSS_TAG}" \
  TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER}" \
  EVAL_MODE="${EVAL_MODE}" \
  SHUFFLE_TRAIN="${SHUFFLE_TRAIN}" \
  MODEL_DROPOUT="${MODEL_DROPOUT}" \
  FEDERATED_ALGORITHMS="${BASE_ALGOS}" \
  bash scripts/run_suite.sh --modes "${SUITE_MODES}"
}

run_ega_case() {
  [[ "${HAS_FEDERATED_MODES}" == true ]] || return 0
  BASE_OUTPUT="${COMMON_BASE_OUTPUT}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  SUITE_SEED="${SUITE_SEED}" \
  RUNTIME_DEVICE="${RUNTIME_DEVICE}" \
  RUN_TAG="${RUN_TAG}" \
  TRACKING_TAG="${TRACKING_TAG}" \
  RUN_CENTRALIZED=false \
  ATTACK_ENABLED="${ATTACK_ENABLED}" \
  ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS}" \
  LOSS_NAME="${LOSS_NAME}" \
  LOSS_TAG="${LOSS_TAG}" \
  TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER}" \
  EVAL_MODE="${EVAL_MODE}" \
  SHUFFLE_TRAIN="${SHUFFLE_TRAIN}" \
  MODEL_DROPOUT="${MODEL_DROPOUT}" \
  FEDERATED_ALGORITHMS=ega \
  EGA_PRETRAIN_DEVICE="${RUNTIME_DEVICE}" \
  "$@" \
  bash scripts/run_suite.sh --modes "${MAPPED_MODES}"
}

run_ega_matrix() {
  if [[ "${LOSS_NAME}" == "mse" ]]; then
    run_ega_case \
      EGA_ENCODED_DIM=160 \
      EGA_HIDDEN_DIM=1024 \
      EGA_RESIDUAL_BLOCKS=2 \
      EGA_QUANTIZATION_LEVEL=159 \
      EGA_NORMALIZATION_EMA=0.95 \
      EGA_PRETRAIN_EPOCHS=150 \
      EGA_PRETRAIN_PATIENCE=30 \
      EGA_PRETRAIN_LR=0.0003

    run_ega_case \
      EGA_ENCODED_DIM=168 \
      EGA_HIDDEN_DIM=2048 \
      EGA_RESIDUAL_BLOCKS=4 \
      EGA_QUANTIZATION_LEVEL=159 \
      EGA_NORMALIZATION_EMA=0.98 \
      EGA_PRETRAIN_EPOCHS=220 \
      EGA_PRETRAIN_PATIENCE=44 \
      EGA_PRETRAIN_LR=0.0002 \
      EGA_PRETRAIN_TRAIN_GROUPS=50000 \
      EGA_PRETRAIN_VAL_GROUPS=25000
  else
    run_ega_case \
      EGA_ENCODED_DIM=160 \
      EGA_HIDDEN_DIM=1536 \
      EGA_RESIDUAL_BLOCKS=3 \
      EGA_QUANTIZATION_LEVEL=127 \
      EGA_NORMALIZATION_EMA=0.95 \
      EGA_PRETRAIN_EPOCHS=150 \
      EGA_PRETRAIN_PATIENCE=30 \
      EGA_PRETRAIN_LR=0.0003

    run_ega_case \
      EGA_ENCODED_DIM=160 \
      EGA_HIDDEN_DIM=2048 \
      EGA_RESIDUAL_BLOCKS=3 \
      EGA_QUANTIZATION_LEVEL=159 \
      EGA_NORMALIZATION_EMA=0.95 \
      EGA_PRETRAIN_EPOCHS=200 \
      EGA_PRETRAIN_PATIENCE=40 \
      EGA_PRETRAIN_LR=0.0002

    run_ega_case \
      EGA_ENCODED_DIM=168 \
      EGA_HIDDEN_DIM=1536 \
      EGA_RESIDUAL_BLOCKS=3 \
      EGA_QUANTIZATION_LEVEL=127 \
      EGA_NORMALIZATION_EMA=0.98 \
      EGA_PRETRAIN_EPOCHS=180 \
      EGA_PRETRAIN_PATIENCE=36 \
      EGA_PRETRAIN_LR=0.00025 \
      EGA_PRETRAIN_TRAIN_GROUPS=40000 \
      EGA_PRETRAIN_VAL_GROUPS=20000
  fi
}

run_base_suite
run_ega_matrix
