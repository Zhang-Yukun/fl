#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
SUITE_SEED="${SUITE_SEED:-2026}"
RUN_TAG="${RUN_TAG:-oracle_attackfreq5}"
TRACKING_TAG="${TRACKING_TAG:-oracle-attackfreq5}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/${RUN_TAG}_seed${SUITE_SEED}_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-fl-${TRACKING_TAG}-v1}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
BASE_PORT="${BASE_PORT:-58000}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"
POLL_SECONDS="${POLL_SECONDS:-1.0}"
RUN_CENTRALIZED="${RUN_CENTRALIZED:-true}"
ROUNDS="${ROUNDS:-}"
PATIENCE="${PATIENCE:-500}"
LOSS_NAME="${LOSS_NAME:-mse}"
LOSS_TAG="${LOSS_TAG:-${LOSS_NAME}}"
ATTACK_ENABLED="${ATTACK_ENABLED:-true}"
ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS:-10}"
ATTACK_CLIENT_SELECTION="${ATTACK_CLIENT_SELECTION:-}"
ATTACK_CLIENTS_PER_ROUND="${ATTACK_CLIENTS_PER_ROUND:-}"
ATTACK_MAX_SAMPLES="${ATTACK_MAX_SAMPLES:-}"
ATTACK_SEED="${ATTACK_SEED:-${SUITE_SEED}}"
TRAIN_LR="${TRAIN_LR:-}"
TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-}"
TRAIN_MOMENTUM="${TRAIN_MOMENTUM:-}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-}"
TRAIN_OPTIMIZER_EPS="${TRAIN_OPTIMIZER_EPS:-}"
SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-true}"
MODEL_DROPOUT="${MODEL_DROPOUT:-0.1}"
ATTACK_LR="${ATTACK_LR:-}"
ATTACK_OPTIMIZER="${ATTACK_OPTIMIZER:-}"
FEDERATED_ALGORITHMS="${FEDERATED_ALGORITHMS:-fedavg,topk,ega}"
TASK_SET="${TASK_SET:-rare}"
TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS:-rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10}"
TASK_CLIENT_IDS="${TASK_CLIENT_IDS:-rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3}"
TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS:-rare}"
TOPK_FRACTION="${TOPK_FRACTION:-}"
QSGD_LEVELS="${QSGD_LEVELS:-}"
QSGD_SEED="${QSGD_SEED:-${SUITE_SEED}}"
RANDOMK_FRACTION="${RANDOMK_FRACTION:-}"
RANDOMK_SEED="${RANDOMK_SEED:-${SUITE_SEED}}"
ADAPTIVE_RDP_SEED="${ADAPTIVE_RDP_SEED:-${SUITE_SEED}}"
QINT8_DTYPE="${QINT8_DTYPE:-}"
QINT8_STOCHASTIC_ROUNDING="${QINT8_STOCHASTIC_ROUNDING:-}"
QINT8_SEED="${QINT8_SEED:-${SUITE_SEED}}"
QINT8_NOISE_MULTIPLIER="${QINT8_NOISE_MULTIPLIER:-}"
EGA_TRACKING_LABEL="${EGA_TRACKING_LABEL:-}"
EGA_ARTIFACT_PATH="${EGA_ARTIFACT_PATH:-}"
EGA_ENCODED_DIM="${EGA_ENCODED_DIM:-240}"
EGA_HIDDEN_DIM="${EGA_HIDDEN_DIM:-1024}"
EGA_RESIDUAL_BLOCKS="${EGA_RESIDUAL_BLOCKS:-2}"
EGA_QUANTIZATION_LEVEL="${EGA_QUANTIZATION_LEVEL:-127}"
EGA_NORMALIZATION="${EGA_NORMALIZATION:-}"
EGA_INITIAL_NORMALIZATION="${EGA_INITIAL_NORMALIZATION:-}"
EGA_MIN_NORMALIZATION="${EGA_MIN_NORMALIZATION:-}"
EGA_NORMALIZATION_STRATEGY="${EGA_NORMALIZATION_STRATEGY:-}"
EGA_NORMALIZATION_EMA="${EGA_NORMALIZATION_EMA:-0.9}"
EGA_ENCODED_DTYPE="${EGA_ENCODED_DTYPE:-}"
EGA_ENCODED_STOCHASTIC_ROUNDING="${EGA_ENCODED_STOCHASTIC_ROUNDING:-}"
EGA_ENCODED_NOISE_STD="${EGA_ENCODED_NOISE_STD:-}"
EGA_ERROR_FEEDBACK="${EGA_ERROR_FEEDBACK:-}"
EGA_QUANTIZATION_SEED="${EGA_QUANTIZATION_SEED:-${SUITE_SEED}}"
EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE:-}"
EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS:-100}"
EGA_PRETRAIN_PATIENCE="${EGA_PRETRAIN_PATIENCE:-}"
EGA_PRETRAIN_MIN_DELTA="${EGA_PRETRAIN_MIN_DELTA:-}"
EGA_PRETRAIN_BATCH_SIZE="${EGA_PRETRAIN_BATCH_SIZE:-}"
EGA_PRETRAIN_LR="${EGA_PRETRAIN_LR:-}"
EGA_PRETRAIN_TRAIN_GROUPS="${EGA_PRETRAIN_TRAIN_GROUPS:-}"
EGA_PRETRAIN_VAL_GROUPS="${EGA_PRETRAIN_VAL_GROUPS:-}"
EGA_PRETRAIN_SEED="${EGA_PRETRAIN_SEED:-${SUITE_SEED}}"

SELECT_MODES="${SELECT_MODES:-}"

usage() {
  cat <<'USAGE'
Usage: bash SCRIPT [--modes centralized,single_sync,single_async,grpc_sync,grpc_async] [--tasks task1,task2|all] [--algorithms fedavg,topk,ega]

Examples:
  bash SCRIPT --modes single_sync
  TRAIN_OPTIMIZER=adam TRAIN_LR=0.001 bash SCRIPT --modes centralized,single_sync
  SUITE_SEED=42 bash SCRIPT --modes single_sync
  SHUFFLE_TRAIN=true MODEL_DROPOUT=0.1 bash SCRIPT --modes single_sync
  TASK_SET=rare,mnist,cifar10 bash SCRIPT --modes single_sync
  TASK_CONFIG_DIRS="rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10" bash SCRIPT --modes single_sync
  FEDERATED_ALGORITHMS=fedavg,ega bash SCRIPT --modes single_sync
  RUNTIME_DEVICE=cuda:1 bash SCRIPT --modes single_sync
  RUN_NAME_PREFIX=debug_ RUN_NAME_SUFFIX=_trial1 bash SCRIPT --modes single_sync
  EGA_PRETRAIN_DEVICE=cuda:1 bash SCRIPT --modes single_sync
  SELECT_MODES=single_async,grpc_async bash SCRIPT
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --modes)
        if [[ $# -lt 2 ]]; then
          echo "--modes requires a comma-separated value" >&2
          exit 1
        fi
        SELECT_MODES="$2"
        shift 2
        ;;
      --tasks)
        if [[ $# -lt 2 ]]; then
          echo "--tasks requires a comma-separated value" >&2
          exit 1
        fi
        TASK_SET="$2"
        shift 2
        ;;
      --algorithms)
        if [[ $# -lt 2 ]]; then
          echo "--algorithms requires a comma-separated value" >&2
          exit 1
        fi
        FEDERATED_ALGORITHMS="$2"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
}

mode_enabled() {
  local target="$1"
  if [[ -z "${SELECT_MODES}" ]]; then
    return 0
  fi
  local raw
  IFS=',' read -r -a raw <<< "${SELECT_MODES}"
  local mode
  for mode in "${raw[@]}"; do
    mode="${mode// /}"
    if [[ -z "${mode}" ]]; then
      continue
    fi
    if [[ "${mode}" == "all" || "${mode}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

algo_enabled() {
  local target="$1"
  if [[ -z "${FEDERATED_ALGORITHMS}" ]]; then
    return 0
  fi
  local raw
  IFS=',' read -r -a raw <<< "${FEDERATED_ALGORITHMS}"
  local algorithm
  for algorithm in "${raw[@]}"; do
    algorithm="${algorithm// /}"
    if [[ -z "${algorithm}" ]]; then
      continue
    fi
    if [[ "${algorithm}" == "all" || "${algorithm}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

parse_named_map_keys() {
  local raw="$1"
  local pair
  IFS=';' read -r -a pairs <<< "${raw}"
  for pair in "${pairs[@]}"; do
    pair="${pair// /}"
    [[ -z "${pair}" ]] && continue
    if [[ "${pair}" != *=* ]]; then
      echo "Invalid map entry: ${pair}" >&2
      exit 1
    fi
    printf '%s\n' "${pair%%=*}"
  done
}

lookup_named_map_value() {
  local raw="$1"
  local key="$2"
  local pair
  IFS=';' read -r -a pairs <<< "${raw}"
  for pair in "${pairs[@]}"; do
    pair="${pair// /}"
    [[ -z "${pair}" ]] && continue
    if [[ "${pair}" != *=* ]]; then
      echo "Invalid map entry: ${pair}" >&2
      exit 1
    fi
    if [[ "${pair%%=*}" == "${key}" ]]; then
      printf '%s\n' "${pair#*=}"
      return 0
    fi
  done
  return 1
}

list_named_values() {
  local raw="$1"
  raw="${raw//,/ }"
  printf '%s\n' ${raw}
}

selected_tasks() {
  local raw="${1:-rare}"
  local -a available_tasks=()
  local -a selected=()
  mapfile -t available_tasks < <(parse_named_map_keys "${TASK_CONFIG_DIRS}")
  local item
  IFS=',' read -r -a parts <<< "${raw}"
  for item in "${parts[@]}"; do
    item="${item// /}"
    [[ -z "${item}" ]] && continue
    if [[ "${item}" == "all" ]]; then
      printf '%s\n' "${available_tasks[@]}"
      return
    fi
    if ! lookup_named_map_value "${TASK_CONFIG_DIRS}" "${item}" >/dev/null; then
      echo "Unsupported TASK_SET entry: ${item}" >&2
      exit 1
    fi
    selected+=("${item}")
  done

  if [[ ${#selected[@]} -eq 0 ]]; then
    echo "TASK_SET must select at least one task." >&2
    exit 1
  fi

  printf '%s\n' "${selected[@]}"
}

task_config_path() {
  local task="$1"
  local config_name="$2"
  local config_dir
  config_dir="$(lookup_named_map_value "${TASK_CONFIG_DIRS}" "${task}")" || {
    echo "Unsupported task=${task}" >&2
    exit 1
  }
  printf '%s/%s.yaml\n' "${config_dir}" "${config_name}"
}

task_client_ids() {
  local task="$1"
  local clients_raw
  clients_raw="$(lookup_named_map_value "${TASK_CLIENT_IDS}" "${task}")" || {
    echo "Missing TASK_CLIENT_IDS entry for task=${task}" >&2
    exit 1
  }
  list_named_values "${clients_raw}"
}

task_uses_loss_override() {
  local task="$1"
  local enabled_task
  while IFS= read -r enabled_task; do
    [[ -z "${enabled_task}" ]] && continue
    if [[ "${enabled_task}" == "${task}" ]]; then
      return 0
    fi
  done < <(list_named_values "${TASK_LOSS_OVERRIDE_TASKS}")
  return 1
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

effective_run_name() {
  local base_name="$1"
  printf '%s%s_seed%s%s\n' "${RUN_NAME_PREFIX}" "${base_name}" "${SUITE_SEED}" "${RUN_NAME_SUFFIX}"
}

effective_tracking_name() {
  local base_name="$1"
  printf '%s-seed%s\n' "${base_name}" "${SUITE_SEED}"
}

ega_name_signature() {
  local encoded_dim="${EGA_ENCODED_DIM}"
  local hidden_dim="${EGA_HIDDEN_DIM}"
  local residual_blocks="${EGA_RESIDUAL_BLOCKS}"
  local quantization_level="${EGA_QUANTIZATION_LEVEL}"
  local normalization_ema="${EGA_NORMALIZATION_EMA}"
  local pretrain_epochs="${EGA_PRETRAIN_EPOCHS}"
  local ema_token="${normalization_ema//./}"
  printf 'ed%s_hd%s_rb%s_q%s_ema%s_pt%s\n' "${encoded_dim}" "${hidden_dim}" "${residual_blocks}" "${quantization_level}" "${ema_token}" "${pretrain_epochs}"
}

effective_ega_label() {
  if [[ -n "${EGA_TRACKING_LABEL}" ]]; then
    printf '%s\n' "${EGA_TRACKING_LABEL}"
    return
  fi
  printf 'ega_%s\n' "$(ega_name_signature)"
}

cleanup_pids() {
  local pids=("$@")
  local pid
  for pid in "${pids[@]}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}

tracking_args() {
  local tracking_name="$1"
  tracking_name="$(effective_tracking_name "${tracking_name}")"
  printf -- '--override
tracking.enabled=true
--override
tracking.offline=true
--override
tracking.project=%s
--override
tracking.group=%s
--override
tracking.name=%s-%s
' "${PROJECT_NAME}" "$(basename "${BASE_OUTPUT}")" "${tracking_name}" "${LOSS_TAG}"
}

base_runtime_args() {
  printf -- '--override
runtime.device=%s
--override
runtime.seed=%s
--override
runtime.deterministic=true
--override
runtime.num_threads=1
--override
runtime.num_interop_threads=1
--override
data.shuffle_train=%s
--override
model.dropout=%s
' "${RUNTIME_DEVICE}" "${SUITE_SEED}" "${SHUFFLE_TRAIN}" "${MODEL_DROPOUT}"
}

training_args() {
  if [[ -n "${TRAIN_OPTIMIZER}" ]]; then
    printf -- '--override
training.optimizer=%s
' "${TRAIN_OPTIMIZER}"
  fi
  if [[ -n "${TRAIN_LR}" ]]; then
    printf -- '--override
training.lr=%s
' "${TRAIN_LR}"
  fi
  if [[ -n "${TRAIN_MOMENTUM}" ]]; then
    printf -- '--override
training.momentum=%s
' "${TRAIN_MOMENTUM}"
  fi
  if [[ -n "${TRAIN_WEIGHT_DECAY}" ]]; then
    printf -- '--override
training.weight_decay=%s
' "${TRAIN_WEIGHT_DECAY}"
  fi
  if [[ -n "${TRAIN_OPTIMIZER_EPS}" ]]; then
    printf -- '--override
training.optimizer_eps=%s
' "${TRAIN_OPTIMIZER_EPS}"
  fi
}

emit_optional_override() {
  local key="$1"
  local value="${2:-}"
  if [[ -n "${value}" ]]; then
    printf -- '--override
%s=%s
' "${key}" "${value}"
  fi
}

fedavg_args() {
  printf -- '--override
federated.algorithm=fedavg
'
}

topk_args() {
  printf -- '--override
federated.algorithm=sparse_fedavg
'
  emit_optional_override 'federated.topk_fraction' "${TOPK_FRACTION}"
}

qsgd_args() {
  printf -- '--override
federated.algorithm=qsgd_fedavg
'
  emit_optional_override 'federated.qsgd_levels' "${QSGD_LEVELS}"
  emit_optional_override 'federated.quantization_seed' "${QSGD_SEED}"
}

randomk_args() {
  printf -- '--override
federated.algorithm=randomk_fedavg
'
  emit_optional_override 'federated.topk_fraction' "${RANDOMK_FRACTION}"
  emit_optional_override 'federated.randomk_seed' "${RANDOMK_SEED}"
}

sign_args() {
  printf -- '--override
federated.algorithm=sign_fedavg
'
}

adaptive_args() {
  printf -- '--override
federated.algorithm=adaptive_clipped_rdp_fedavg
'
  emit_optional_override 'adaptive_clipped_rdp.seed' "${ADAPTIVE_RDP_SEED}"
}

qint8_args() {
  printf -- '--override
federated.algorithm=secure_quantized_fedavg
'
  emit_optional_override 'federated.quantization_dtype' "${QINT8_DTYPE}"
  emit_optional_override 'federated.quantization_stochastic_rounding' "${QINT8_STOCHASTIC_ROUNDING}"
  emit_optional_override 'federated.quantization_seed' "${QINT8_SEED}"
  emit_optional_override 'privacy.noise_multiplier' "${QINT8_NOISE_MULTIPLIER}"
}

ega_args() {
  printf -- '--override
federated.algorithm=ega_fedavg
'
  emit_optional_override 'federated.quantization_seed' "${EGA_QUANTIZATION_SEED}"
  emit_optional_override 'ega.artifact_path' "${EGA_ARTIFACT_PATH}"
  emit_optional_override 'ega.encoded_dim' "${EGA_ENCODED_DIM}"
  emit_optional_override 'ega.hidden_dim' "${EGA_HIDDEN_DIM}"
  emit_optional_override 'ega.residual_blocks' "${EGA_RESIDUAL_BLOCKS}"
  emit_optional_override 'ega.quantization_level' "${EGA_QUANTIZATION_LEVEL}"
  emit_optional_override 'ega.normalization' "${EGA_NORMALIZATION}"
  emit_optional_override 'ega.initial_normalization' "${EGA_INITIAL_NORMALIZATION}"
  emit_optional_override 'ega.min_normalization' "${EGA_MIN_NORMALIZATION}"
  emit_optional_override 'ega.normalization_strategy' "${EGA_NORMALIZATION_STRATEGY}"
  emit_optional_override 'ega.normalization_ema' "${EGA_NORMALIZATION_EMA}"
  emit_optional_override 'ega.encoded_dtype' "${EGA_ENCODED_DTYPE}"
  emit_optional_override 'ega.encoded_stochastic_rounding' "${EGA_ENCODED_STOCHASTIC_ROUNDING}"
  emit_optional_override 'ega.encoded_noise_std' "${EGA_ENCODED_NOISE_STD}"
  emit_optional_override 'ega.error_feedback' "${EGA_ERROR_FEEDBACK}"
  emit_optional_override 'ega.pretrain.device' "${EGA_PRETRAIN_DEVICE}"
  emit_optional_override 'ega.pretrain.epochs' "${EGA_PRETRAIN_EPOCHS}"
  emit_optional_override 'ega.pretrain.patience' "${EGA_PRETRAIN_PATIENCE}"
  emit_optional_override 'ega.pretrain.min_delta' "${EGA_PRETRAIN_MIN_DELTA}"
  emit_optional_override 'ega.pretrain.batch_size' "${EGA_PRETRAIN_BATCH_SIZE}"
  emit_optional_override 'ega.pretrain.lr' "${EGA_PRETRAIN_LR}"
  emit_optional_override 'ega.pretrain.train_groups' "${EGA_PRETRAIN_TRAIN_GROUPS}"
  emit_optional_override 'ega.pretrain.val_groups' "${EGA_PRETRAIN_VAL_GROUPS}"
  emit_optional_override 'ega.pretrain.seed' "${EGA_PRETRAIN_SEED}"
}

centralized_round_args() {
  local task="$1"
  if [[ -n "${ROUNDS}" ]]; then
    printf -- '--override
training.epochs=%s
' "${ROUNDS}"
  fi
  if [[ -n "${PATIENCE}" ]]; then
    printf -- '--override
training.patience=%s
--override
training.min_delta=0.0
' "${PATIENCE}"
  fi
  while IFS= read -r line; do
    printf '%s\n' "${line}"
  done < <(training_args)
  if task_uses_loss_override "${task}"; then
    printf -- '--override
training.loss=%s
' "${LOSS_NAME}"
  fi
}

federated_common_args() {
  local task="$1"
  printf -- '--override
experiment.mode=federated
--override
--override
--override
attack.enabled=%s
--override
attack.async_enabled=false
' "${ATTACK_ENABLED}"
  if [[ "${ATTACK_ENABLED}" == "true" ]]; then
    printf -- '--override
attack.frequency_rounds=%s
' "${ATTACK_FREQUENCY_ROUNDS}"
    if [[ -n "${ATTACK_CLIENT_SELECTION}" ]]; then
      printf -- '--override
attack.client_selection=%s
' "${ATTACK_CLIENT_SELECTION}"
    fi
    if [[ -n "${ATTACK_CLIENTS_PER_ROUND}" ]]; then
      printf -- '--override
attack.clients_per_round=%s
' "${ATTACK_CLIENTS_PER_ROUND}"
    fi
    if [[ -n "${ATTACK_MAX_SAMPLES}" ]]; then
      printf -- '--override
attack.max_samples=%s
' "${ATTACK_MAX_SAMPLES}"
    fi
    if [[ -n "${ATTACK_SEED}" ]]; then
      printf -- '--override
attack.seed=%s
' "${ATTACK_SEED}"
    fi
    if [[ -n "${ATTACK_LR}" ]]; then
      printf -- '--override
attack.lr=%s
' "${ATTACK_LR}"
    fi
    if [[ -n "${ATTACK_OPTIMIZER}" ]]; then
      printf -- '--override
attack.optimizer=%s
' "${ATTACK_OPTIMIZER}"
    fi
  fi
  if [[ -n "${ROUNDS}" ]]; then
    printf -- '--override
federated.rounds=%s
' "${ROUNDS}"
  fi
  if [[ -n "${PATIENCE}" ]]; then
    printf -- '--override
training.patience=%s
--override
training.min_delta=0.0
' "${PATIENCE}"
  fi
  while IFS= read -r line; do
    printf '%s\n' "${line}"
  done < <(training_args)
  if task_uses_loss_override "${task}"; then
    printf -- '--override
training.loss=%s
' "${LOSS_NAME}"
  fi
}

run_single() {
  local task="$1"
  local run_name="$2"
  local tracking_name="$3"
  local config="$4"
  shift 4
  run_name="$(effective_run_name "${run_name}")"
  local outdir="${BASE_OUTPUT}/${task}/${run_name}"
  local -a cmd=(
    "${PYTHON_BIN}" -m fedlab.entrypoints.train
    --config "${config}"
    --mode federated
    --override "experiment.output_dir=${outdir}"
    --override "experiment.name=${run_name}"
  )
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(tracking_args "${task}-${tracking_name}")
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(base_runtime_args)
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(federated_common_args "${task}")
  cmd+=("$@")

  log "starting ${run_name}"
  PYTHONPATH=. "${cmd[@]}"
  log "finished ${run_name}"
}

run_grpc() {
  local task="$1"
  local run_name="$2"
  local tracking_name="$3"
  local config="$4"
  local port="$5"
  shift 5
  run_name="$(effective_run_name "${run_name}")"
  local outdir="${BASE_OUTPUT}/${task}/${run_name}"
  local address="127.0.0.1:${port}"
  local server_pid=""
  local client_pids=()

  mkdir -p "${outdir}"

  local -a server_cmd=(
    "${PYTHON_BIN}" -m fedlab.entrypoints.server
    --config "${config}"
    --override "experiment.output_dir=${outdir}"
    --override "experiment.name=${run_name}"
    --override "experiment.mode=federated"
    --override "grpc.address=${address}"
    --override "grpc.server_address=${address}"
    --override "grpc.poll_seconds=${POLL_SECONDS}"
  )
  while IFS= read -r line; do
    server_cmd+=("${line}")
  done < <(tracking_args "${task}-${tracking_name}")
  while IFS= read -r line; do
    server_cmd+=("${line}")
  done < <(base_runtime_args)
  while IFS= read -r line; do
    server_cmd+=("${line}")
  done < <(federated_common_args "${task}")
  server_cmd+=("$@")

  log "starting ${run_name} on ${address}"
  PYTHONPATH=. "${server_cmd[@]}" > "${outdir}/server.log" 2>&1 &
  server_pid=$!
  sleep "${STARTUP_WAIT_SECONDS}"

  local -a client_ids=()
  mapfile -t client_ids < <(task_client_ids "${task}")
  local client_id
  for client_id in "${client_ids[@]}"; do
    local -a client_cmd=(
      "${PYTHON_BIN}" -m fedlab.entrypoints.client
      --client-id "${client_id}"
      --config "${config}"
      --override "experiment.output_dir=${outdir}"
      --override "experiment.name=${run_name}-${client_id}"
      --override "experiment.mode=federated"
      --override "grpc.address=${address}"
      --override "grpc.server_address=${address}"
      --override "grpc.poll_seconds=${POLL_SECONDS}"
      --override tracking.enabled=false
    )
    while IFS= read -r line; do
      client_cmd+=("${line}")
    done < <(base_runtime_args)
    while IFS= read -r line; do
      client_cmd+=("${line}")
    done < <(federated_common_args "${task}")
    client_cmd+=("$@")

    PYTHONPATH=. "${client_cmd[@]}" > "${outdir}/client_${client_id}.log" 2>&1 &
    client_pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${client_pids[@]}"; do
    if ! wait "${pid}"; then
      status=$?
      break
    fi
  done

  if [[ "${status}" -ne 0 ]]; then
    log "${run_name} failed in a client process, cleaning up"
    cleanup_pids "${server_pid}" "${client_pids[@]}"
    wait "${server_pid}" 2>/dev/null || true
    return "${status}"
  fi

  if ! wait "${server_pid}"; then
    status=$?
    log "${run_name} failed in the server process"
    cleanup_pids "${client_pids[@]}"
    return "${status}"
  fi

  log "finished ${run_name}"
}

run_centralized() {
  local task="$1"
  local run_name="$2"
  local tracking_name="$3"
  local config="$4"
  shift 4
  run_name="$(effective_run_name "${run_name}")"
  local outdir="${BASE_OUTPUT}/${task}/${run_name}"
  local -a cmd=(
    "${PYTHON_BIN}" -m fedlab.entrypoints.train
    --config "${config}"
    --mode centralized
    --override "experiment.output_dir=${outdir}"
    --override "experiment.name=${run_name}"
    --override "experiment.mode=centralized"
  )
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(tracking_args "${task}-${tracking_name}")
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(base_runtime_args)
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(centralized_round_args "${task}")
  cmd+=("$@")

  log "starting ${run_name}"
  PYTHONPATH=. "${cmd[@]}"
  log "finished ${run_name}"
}


main() {
  mkdir -p "${BASE_OUTPUT}"
  log "output root: ${BASE_OUTPUT}"
  log "device=${RUNTIME_DEVICE} project=${PROJECT_NAME}"
  log "tasks=${TASK_SET} federated_algorithms=${FEDERATED_ALGORITHMS}"

  local -a TASK_LIST=()
  mapfile -t TASK_LIST < <(selected_tasks "${TASK_SET}")

  local task
  for task in "${TASK_LIST[@]}"; do
    log "starting task=${task}"

    if [[ "${RUN_CENTRALIZED}" == "true" ]] && mode_enabled centralized; then
      run_centralized         "${task}"         centralized_uupdate_dmodel_${RUN_TAG}         centralized-${TRACKING_TAG}         "$(task_config_path "${task}" centralized)"
    fi

    local -a modes=(single_sync single_async grpc_sync grpc_async)
    local port="${BASE_PORT}"
    local mode
    for mode in "${modes[@]}"; do
      if ! mode_enabled "${mode}"; then
        continue
      fi
      if [[ "${mode}" == grpc_* ]]; then
        if algo_enabled fedavg; then
          local -a fedavg_override_args=()
          mapfile -t fedavg_override_args < <(fedavg_args)
          run_grpc "${task}" "fedavg_${mode}_uupdate_dmodel_${RUN_TAG}" "fedavg-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${port}" "${fedavg_override_args[@]}"
          port=$((port + 1))
        fi

        if algo_enabled ega; then
          local ega_label="$(effective_ega_label)"
          local ega_run_name="${ega_label}_${mode}_uupdate_dmodel_${RUN_TAG}"
          local ega_tracking_label="${ega_label//_/-}"
          local -a ega_override_args=()
          mapfile -t ega_override_args < <(ega_args)
          run_grpc "${task}" "${ega_run_name}" "${ega_tracking_label}-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" ega)" "${port}" "${ega_override_args[@]}"
          port=$((port + 1))
        fi

        if algo_enabled topk; then
          local -a topk_override_args=()
          mapfile -t topk_override_args < <(topk_args)
          run_grpc "${task}" "topk_${mode}_uupdate_dmodel_${RUN_TAG}" "topk-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" topk)" "${port}" "${topk_override_args[@]}"
          port=$((port + 1))
        fi

        if algo_enabled qsgd; then
          local -a qsgd_override_args=()
          mapfile -t qsgd_override_args < <(qsgd_args)
          run_grpc "${task}" "qsgd_${mode}_uupdate_dmodel_${RUN_TAG}" "qsgd-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${port}" "${qsgd_override_args[@]}"
          port=$((port + 1))
        fi

        if algo_enabled randomk; then
          local -a randomk_override_args=()
          mapfile -t randomk_override_args < <(randomk_args)
          run_grpc "${task}" "randomk_${mode}_uupdate_dmodel_${RUN_TAG}" "randomk-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${port}" "${randomk_override_args[@]}"
          port=$((port + 1))
        fi

        if algo_enabled sign; then
          local -a sign_override_args=()
          mapfile -t sign_override_args < <(sign_args)
          run_grpc "${task}" "sign_${mode}_uupdate_dmodel_${RUN_TAG}" "sign-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${port}" "${sign_override_args[@]}"
          port=$((port + 1))
        fi

        if algo_enabled adaptive; then
          local -a adaptive_override_args=()
          mapfile -t adaptive_override_args < <(adaptive_args)
          run_grpc "${task}" "adaptive_${mode}_uupdate_dmodel_${RUN_TAG}" "adaptive-rdp-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${port}" "${adaptive_override_args[@]}"
          port=$((port + 1))
        fi

        if algo_enabled qint8; then
          local -a qint8_override_args=()
          mapfile -t qint8_override_args < <(qint8_args)
          run_grpc "${task}" "qint8_${mode}_uupdate_dmodel_${RUN_TAG}" "secure-quantized-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${port}" "${qint8_override_args[@]}"
          port=$((port + 1))
        fi
      else
        local async_flag=false
        local -a async_workers=()
        if [[ "${mode}" == "single_async" ]]; then
          async_flag=true
          async_workers=(--override attack.async_workers=1)
        fi

        if algo_enabled fedavg; then
          local -a fedavg_override_args=()
          mapfile -t fedavg_override_args < <(fedavg_args)
          run_single "${task}" "fedavg_${mode}_uupdate_dmodel_${RUN_TAG}" "fedavg-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${fedavg_override_args[@]}" --override attack.async_enabled=${async_flag} "${async_workers[@]}"
        fi

        if algo_enabled ega; then
          local ega_label="$(effective_ega_label)"
          local ega_run_name="${ega_label}_${mode}_uupdate_dmodel_${RUN_TAG}"
          local ega_tracking_label="${ega_label//_/-}"
          local -a ega_override_args=()
          mapfile -t ega_override_args < <(ega_args)
          run_single "${task}" "${ega_run_name}" "${ega_tracking_label}-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" ega)" "${ega_override_args[@]}" --override attack.async_enabled=${async_flag} "${async_workers[@]}"
        fi

        if algo_enabled topk; then
          local -a topk_override_args=()
          mapfile -t topk_override_args < <(topk_args)
          run_single "${task}" "topk_${mode}_uupdate_dmodel_${RUN_TAG}" "topk-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" topk)" "${topk_override_args[@]}" --override attack.async_enabled=${async_flag} "${async_workers[@]}"
        fi

        if algo_enabled qsgd; then
          local -a qsgd_override_args=()
          mapfile -t qsgd_override_args < <(qsgd_args)
          run_single "${task}" "qsgd_${mode}_uupdate_dmodel_${RUN_TAG}" "qsgd-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${qsgd_override_args[@]}" --override attack.async_enabled=${async_flag} "${async_workers[@]}"
        fi

        if algo_enabled randomk; then
          local -a randomk_override_args=()
          mapfile -t randomk_override_args < <(randomk_args)
          run_single "${task}" "randomk_${mode}_uupdate_dmodel_${RUN_TAG}" "randomk-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${randomk_override_args[@]}" --override attack.async_enabled=${async_flag} "${async_workers[@]}"
        fi

        if algo_enabled sign; then
          local -a sign_override_args=()
          mapfile -t sign_override_args < <(sign_args)
          run_single "${task}" "sign_${mode}_uupdate_dmodel_${RUN_TAG}" "sign-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${sign_override_args[@]}" --override attack.async_enabled=${async_flag} "${async_workers[@]}"
        fi

        if algo_enabled adaptive; then
          local -a adaptive_override_args=()
          mapfile -t adaptive_override_args < <(adaptive_args)
          run_single "${task}" "adaptive_${mode}_uupdate_dmodel_${RUN_TAG}" "adaptive-rdp-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${adaptive_override_args[@]}" --override attack.async_enabled=${async_flag} "${async_workers[@]}"
        fi

        if algo_enabled qint8; then
          local -a qint8_override_args=()
          mapfile -t qint8_override_args < <(qint8_args)
          run_single "${task}" "qint8_${mode}_uupdate_dmodel_${RUN_TAG}" "secure-quantized-${mode}-uupdate-dmodel-${TRACKING_TAG}" "$(task_config_path "${task}" fedavg)" "${qint8_override_args[@]}" --override attack.async_enabled=${async_flag} "${async_workers[@]}"
        fi
      fi
    done
  done

  log "suite finished"
}

parse_args "$@"
main
