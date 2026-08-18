#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
BASE_PORT="${BASE_PORT:-56000}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/formal_suite_$(date +%Y%m%d_%H%M%S)}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"
POLL_SECONDS="${POLL_SECONDS:-1.0}"
SUITE_ROUNDS="${SUITE_ROUNDS:-}"
ATTACK_FREQUENCY="${ATTACK_FREQUENCY:-}"

COMMON_DETERMINISTIC_OVERRIDES=(
  --override runtime.seed=2026
  --override runtime.deterministic=true
  --override runtime.num_threads=1
  --override runtime.num_interop_threads=1
  --override data.shuffle_train=false
  --override model.dropout=0.0
  --override attack.seed=2026
)
QINT8_DETERMINISTIC_OVERRIDES=(
  --override federated.quantization_stochastic_rounding=false
  --override federated.quantization_seed=2026
  --override privacy.noise_multiplier=0.0
)
CENTRALIZED_OVERRIDES=()
FEDERATED_OVERRIDES=()

if [[ -n "${SUITE_ROUNDS}" ]]; then
  CENTRALIZED_OVERRIDES+=(--override "centralized.rounds=${SUITE_ROUNDS}")
  FEDERATED_OVERRIDES+=(--override "federated.rounds=${SUITE_ROUNDS}")
fi

if [[ -n "${ATTACK_FREQUENCY}" ]]; then
  FEDERATED_OVERRIDES+=(--override "attack.frequency_rounds=${ATTACK_FREQUENCY}")
fi

log() {
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*"
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

run_single() {
  local name="$1"
  local config="$2"
  local mode="$3"
  shift 3
  local outdir="${BASE_OUTPUT}/${name}"

  log "starting ${name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.train \
    --config "${config}" \
    --mode "${mode}" \
    --override "experiment.output_dir=${outdir}" \
    --override "runtime.device=${RUNTIME_DEVICE}" \
    "${COMMON_DETERMINISTIC_OVERRIDES[@]}" \
    "$@"
  log "finished ${name}"
}

run_grpc() {
  local name="$1"
  local config="$2"
  local port="$3"
  shift 3

  local outdir="${BASE_OUTPUT}/${name}"
  local address="127.0.0.1:${port}"
  local server_pid=""
  local client_pids=()

  mkdir -p "${outdir}"
  log "starting ${name} on ${address}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.server \
    --config "${config}" \
    --override "experiment.output_dir=${outdir}" \
    --override "runtime.device=${RUNTIME_DEVICE}" \
    --override "grpc.address=${address}" \
    --override "grpc.server_address=${address}" \
    --override "grpc.poll_seconds=${POLL_SECONDS}" \
    "${COMMON_DETERMINISTIC_OVERRIDES[@]}" \
    "$@" \
    > "${outdir}/server.log" 2>&1 &
  server_pid=$!

  sleep "${STARTUP_WAIT_SECONDS}"

  for client_id in Nd2O3 CeO2 La2O3; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.client \
      --client-id "${client_id}" \
      --config "${config}" \
      --override "experiment.output_dir=${outdir}" \
      --override "runtime.device=${RUNTIME_DEVICE}" \
      --override "grpc.address=${address}" \
      --override "grpc.server_address=${address}" \
      --override "grpc.poll_seconds=${POLL_SECONDS}" \
      "${COMMON_DETERMINISTIC_OVERRIDES[@]}" \
      "$@" \
      > "${outdir}/client_${client_id}.log" 2>&1 &
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

main() {
  log "formal suite output root: ${BASE_OUTPUT}"
  mkdir -p "${BASE_OUTPUT}"

  run_single \
    centralized \
    configs/rawdata2_fedavg.yaml \
    centralized \
    --override experiment.mode=centralized \
    "${CENTRALIZED_OVERRIDES[@]}"

  run_single \
    fedavg_single_sync \
    configs/rawdata2_fedavg.yaml \
    federated \
    --override experiment.mode=federated \
    --override federated.algorithm=fedavg \
    --override attack.async_enabled=false \
    "${FEDERATED_OVERRIDES[@]}"

  run_single \
    fedavg_single_async \
    configs/rawdata2_fedavg.yaml \
    federated \
    --override experiment.mode=federated \
    --override federated.algorithm=fedavg \
    --override attack.async_enabled=true \
    --override attack.async_workers=1 \
    --override attack.device=${RUNTIME_DEVICE} \
    "${FEDERATED_OVERRIDES[@]}"

  run_grpc \
    fedavg_grpc_sync \
    configs/rawdata2_fedavg.yaml \
    "${BASE_PORT}" \
    --override experiment.mode=federated \
    --override federated.algorithm=fedavg \
    --override attack.async_enabled=false \
    "${FEDERATED_OVERRIDES[@]}"

  run_grpc \
    fedavg_grpc_async \
    configs/rawdata2_fedavg.yaml \
    "$((BASE_PORT + 1))" \
    --override experiment.mode=federated \
    --override federated.algorithm=fedavg \
    --override attack.async_enabled=true \
    --override attack.async_workers=1 \
    --override attack.device=${RUNTIME_DEVICE} \
    "${FEDERATED_OVERRIDES[@]}"

  run_single \
    qint8_single_sync \
    configs/rawdata2_secure_quantized_fedavg.yaml \
    federated \
    --override experiment.mode=federated \
    --override federated.quantization_dtype=int8 \
    --override attack.async_enabled=false \
    "${QINT8_DETERMINISTIC_OVERRIDES[@]}" \
    "${FEDERATED_OVERRIDES[@]}"

  run_single \
    qint8_single_async \
    configs/rawdata2_secure_quantized_fedavg.yaml \
    federated \
    --override experiment.mode=federated \
    --override federated.quantization_dtype=int8 \
    --override attack.async_enabled=true \
    --override attack.async_workers=1 \
    --override attack.device=${RUNTIME_DEVICE} \
    "${QINT8_DETERMINISTIC_OVERRIDES[@]}" \
    "${FEDERATED_OVERRIDES[@]}"

  run_grpc \
    qint8_grpc_sync \
    configs/rawdata2_secure_quantized_fedavg.yaml \
    "$((BASE_PORT + 2))" \
    --override experiment.mode=federated \
    --override federated.quantization_dtype=int8 \
    --override attack.async_enabled=false \
    "${QINT8_DETERMINISTIC_OVERRIDES[@]}" \
    "${FEDERATED_OVERRIDES[@]}"

  run_grpc \
    qint8_grpc_async \
    configs/rawdata2_secure_quantized_fedavg.yaml \
    "$((BASE_PORT + 3))" \
    --override experiment.mode=federated \
    --override federated.quantization_dtype=int8 \
    --override attack.async_enabled=true \
    --override attack.async_workers=1 \
    --override attack.device=${RUNTIME_DEVICE} \
    "${QINT8_DETERMINISTIC_OVERRIDES[@]}" \
    "${FEDERATED_OVERRIDES[@]}"

  run_single \
    adaptive_clipped_rdp_single_sync \
    configs/rawdata2_adaptive_clipped_rdp_fedavg_deterministic.yaml \
    federated \
    --override experiment.mode=federated \
    --override attack.async_enabled=false \
    "${FEDERATED_OVERRIDES[@]}"

  log "all runs finished"
}

main "$@"
