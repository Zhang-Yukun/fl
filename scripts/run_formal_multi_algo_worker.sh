#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
BASE_OUTPUT="${BASE_OUTPUT:?BASE_OUTPUT is required}"
BASE_PORT="${BASE_PORT:?BASE_PORT is required}"
WORKER_KIND="${WORKER_KIND:?WORKER_KIND is required}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"
POLL_SECONDS="${POLL_SECONDS:-1.0}"
ROUNDS="${ROUNDS:-300}"
ATTACK_FREQUENCY="${ATTACK_FREQUENCY:-5}"
SUITE_TAG="${SUITE_TAG:-formal300_sgd}"
TRAIN_LR="${TRAIN_LR:-0.001}"
ATTACK_LR="${ATTACK_LR:-0.02}"
ATTACK_STEPS="${ATTACK_STEPS:-200}"
PATIENCE="$((ROUNDS + 1))"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [${WORKER_KIND}] $*"
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
  local short_name="$1"
  printf -- '--override
tracking.enabled=true
--override
tracking.offline=true
--override
tracking.project=federated-rare-earth
--override
tracking.name=%s
--override
tracking.group=%s
--override
tracking.job_type=%s
' "${short_name}" "$(basename "${BASE_OUTPUT}")" "${WORKER_KIND}"
}

common_args() {
  printf -- '--override
runtime.seed=2026
--override
runtime.deterministic=true
--override
runtime.num_threads=1
--override
runtime.num_interop_threads=1
--override
runtime.device=%s
--override
training.optimizer=sgd
--override
training.lr=%s
--override
training.momentum=0.0
--override
training.weight_decay=0.0
--override
training.patience=%s
--override
training.min_delta=0.0
--override
attack.seed=2026
--override
attack.steps=%s
--override
attack.lr=%s
--override
attack.local_optimizer=sgd
--override
attack.local_lr=%s
--override
attack.device=%s
--override
attack.frequency_rounds=%s
' "${RUNTIME_DEVICE}" "${TRAIN_LR}" "${PATIENCE}" "${ATTACK_STEPS}" "${ATTACK_LR}" "${TRAIN_LR}" "${RUNTIME_DEVICE}" "${ATTACK_FREQUENCY}"
}

run_single() {
  local name="$1"
  local short_name="$2"
  local config="$3"
  shift 3
  local outdir="${BASE_OUTPUT}/${name}"
  log "starting ${name}"
  local -a cmd=("${PYTHON_BIN}" -m fedlab.entrypoints.train --config "${config}" --mode federated --override "experiment.output_dir=${outdir}" --override "experiment.mode=federated")
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(tracking_args "${short_name}")
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(common_args)
  cmd+=("$@")
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${cmd[@]}"
  log "finished ${name}"
}

run_centralized() {
  local name="$1"
  local short_name="$2"
  local config="$3"
  shift 3
  local outdir="${BASE_OUTPUT}/${name}"
  log "starting ${name}"
  local -a cmd=("${PYTHON_BIN}" -m fedlab.entrypoints.train --config "${config}" --mode centralized --override "experiment.output_dir=${outdir}" --override "experiment.mode=centralized")
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(tracking_args "${short_name}")
  cmd+=(--override "runtime.seed=2026" --override "runtime.deterministic=true" --override "runtime.num_threads=1" --override "runtime.num_interop_threads=1" --override "runtime.device=${RUNTIME_DEVICE}" --override "training.optimizer=sgd" --override "training.lr=${TRAIN_LR}" --override "training.momentum=0.0" --override "training.weight_decay=0.0" --override "centralized.rounds=${ROUNDS}" --override "training.patience=${PATIENCE}" --override "training.min_delta=0.0")
  cmd+=("$@")
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${cmd[@]}"
  log "finished ${name}"
}

run_grpc() {
  local name="$1"
  local short_name="$2"
  local config="$3"
  local port="$4"
  shift 4
  local outdir="${BASE_OUTPUT}/${name}"
  local address="127.0.0.1:${port}"
  local server_pid=""
  local client_pids=()
  mkdir -p "${outdir}"
  log "starting ${name} on ${address}"

  local -a server_cmd=("${PYTHON_BIN}" -m fedlab.entrypoints.server --config "${config}" --override "experiment.output_dir=${outdir}" --override "experiment.mode=federated" --override "grpc.address=${address}" --override "grpc.server_address=${address}" --override "grpc.poll_seconds=${POLL_SECONDS}")
  while IFS= read -r line; do
    server_cmd+=("${line}")
  done < <(tracking_args "${short_name}")
  while IFS= read -r line; do
    server_cmd+=("${line}")
  done < <(common_args)
  server_cmd+=("$@")
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${server_cmd[@]}" > "${outdir}/server.log" 2>&1 &
  server_pid=$!
  sleep "${STARTUP_WAIT_SECONDS}"

  for client_id in Nd2O3 CeO2 La2O3; do
    local -a client_cmd=("${PYTHON_BIN}" -m fedlab.entrypoints.client --client-id "${client_id}" --config "${config}" --override "experiment.output_dir=${outdir}" --override "experiment.mode=federated" --override "grpc.address=${address}" --override "grpc.server_address=${address}" --override "grpc.poll_seconds=${POLL_SECONDS}")
    while IFS= read -r line; do
      client_cmd+=("${line}")
    done < <(tracking_args "${short_name}-${client_id}")
    while IFS= read -r line; do
      client_cmd+=("${line}")
    done < <(common_args)
    client_cmd+=("$@")
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${client_cmd[@]}" > "${outdir}/client_${client_id}.log" 2>&1 &
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
    log "${name} failed in a client process, cleaning up"
    cleanup_pids "${server_pid}" "${client_pids[@]}"
    wait "${server_pid}" 2>/dev/null || true
    return "${status}"
  fi
  if ! wait "${server_pid}"; then
    status=$?
    log "${name} failed in the server process"
    cleanup_pids "${client_pids[@]}"
    return "${status}"
  fi
  log "finished ${name}"
}

run_algo_block() {
  local algo_key="$1"
  local mode_key="$2"
  local config="$3"
  local short_prefix="$4"
  shift 4
  local -a extra=("$@")
  local name="${algo_key}_${mode_key}"
  local short_name="${short_prefix}-${mode_key}"
  if [[ "${mode_key}" == single_sync ]]; then
    run_single "${name}" "${short_name}" "${config}" --override attack.async_enabled=false "${extra[@]}"
  elif [[ "${mode_key}" == single_async ]]; then
    run_single "${name}" "${short_name}" "${config}" --override attack.async_enabled=true --override attack.async_workers=1 "${extra[@]}"
  elif [[ "${mode_key}" == grpc_sync ]]; then
    run_grpc "${name}" "${short_name}" "${config}" "$5" --override attack.async_enabled=false "${extra[@]}"
  else
    run_grpc "${name}" "${short_name}" "${config}" "$5" --override attack.async_enabled=true --override attack.async_workers=1 "${extra[@]}"
  fi
}

main() {
  mkdir -p "${BASE_OUTPUT}"
  log "output root: ${BASE_OUTPUT} gpu=${GPU_ID} device=${RUNTIME_DEVICE}"
  local -a sync_modes=(single_sync grpc_sync)
  local -a async_modes=(single_async grpc_async)
  local -a modes=()
  if [[ "${WORKER_KIND}" == sync ]]; then
    modes=("${sync_modes[@]}")
    run_centralized centralized cen configs/centralized.yaml
  else
    modes=("${async_modes[@]}")
  fi
  local port="${BASE_PORT}"
  local mode
  for mode in "${modes[@]}"; do
    if [[ "${mode}" == grpc_* ]]; then
      run_grpc "fedavg_${mode}" "fa-${mode}" configs/fedavg.yaml "${port}" --override federated.algorithm=fedavg --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
      port=$((port + 1))
    else
      run_single "fedavg_${mode}" "fa-${mode}" configs/fedavg.yaml --override federated.algorithm=fedavg --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
    fi

    if [[ "${mode}" == grpc_* ]]; then
      run_grpc "topk_${mode}" "tk-${mode}" configs/topk.yaml "${port}" --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
      port=$((port + 1))
    else
      run_single "topk_${mode}" "tk-${mode}" configs/topk.yaml --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
    fi

    if [[ "${mode}" == grpc_* ]]; then
      run_grpc "qint8_${mode}" "qi-${mode}" configs/secure_quantized_fedavg.yaml "${port}" --override federated.rounds=${ROUNDS} --override federated.quantization_dtype=int8 --override federated.quantization_stochastic_rounding=false --override privacy.noise_multiplier=0.0 --override data.shuffle_train=false
      port=$((port + 1))
    else
      run_single "qint8_${mode}" "qi-${mode}" configs/secure_quantized_fedavg.yaml --override federated.rounds=${ROUNDS} --override federated.quantization_dtype=int8 --override federated.quantization_stochastic_rounding=false --override privacy.noise_multiplier=0.0 --override data.shuffle_train=false
    fi

    if [[ "${mode}" == grpc_* ]]; then
      run_grpc "randomk_${mode}" "rk-${mode}" configs/randomk.yaml "${port}" --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
      port=$((port + 1))
    else
      run_single "randomk_${mode}" "rk-${mode}" configs/randomk.yaml --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
    fi

    if [[ "${mode}" == grpc_* ]]; then
      run_grpc "sign_${mode}" "sg-${mode}" configs/sign.yaml "${port}" --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
      port=$((port + 1))
    else
      run_single "sign_${mode}" "sg-${mode}" configs/sign.yaml --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
    fi

    if [[ "${mode}" == grpc_* ]]; then
      run_grpc "qsgd_${mode}" "qg-${mode}" configs/qsgd.yaml "${port}" --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
      port=$((port + 1))
    else
      run_single "qsgd_${mode}" "qg-${mode}" configs/qsgd.yaml --override federated.rounds=${ROUNDS} --override data.shuffle_train=false
    fi

    if [[ "${mode}" == grpc_* ]]; then
      run_grpc "adaptive_${mode}" "ac-${mode}" configs/adaptive_clipped_rdp_fedavg.yaml "${port}" --override federated.rounds=${ROUNDS} --override data.shuffle_train=false --override adaptive_clipped_rdp.seed=2026
      port=$((port + 1))
    else
      run_single "adaptive_${mode}" "ac-${mode}" configs/adaptive_clipped_rdp_fedavg.yaml --override federated.rounds=${ROUNDS} --override data.shuffle_train=false --override adaptive_clipped_rdp.seed=2026
    fi
  done
  log "worker finished"
}

main "$@"
