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

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
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
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m federated_ts.entrypoints.train     --config "${config}"     --mode "${mode}"     --override "experiment.output_dir=${outdir}"     --override "runtime.device=${RUNTIME_DEVICE}"     "$@"
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

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m federated_ts.entrypoints.server     --config "${config}"     --override "experiment.output_dir=${outdir}"     --override "runtime.device=${RUNTIME_DEVICE}"     --override "grpc.address=${address}"     --override "grpc.server_address=${address}"     --override "grpc.poll_seconds=${POLL_SECONDS}"     "$@"     > "${outdir}/server.log" 2>&1 &
  server_pid=$!

  sleep "${STARTUP_WAIT_SECONDS}"

  for client_id in Nd2O3 CeO2 La2O3; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m federated_ts.entrypoints.client       --client-id "${client_id}"       --config "${config}"       --override "experiment.output_dir=${outdir}"       --override "runtime.device=${RUNTIME_DEVICE}"       --override "grpc.address=${address}"       --override "grpc.server_address=${address}"       --override "grpc.poll_seconds=${POLL_SECONDS}"       "$@"       > "${outdir}/client_${client_id}.log" 2>&1 &
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

  run_single     centralized     configs/rawdata2_patchtst.yaml     centralized     --override experiment.mode=centralized

  run_single     fedavg_single_sync     configs/rawdata2_patchtst.yaml     federated     --override experiment.mode=federated     --override federated.algorithm=fedavg     --override attack.async_enabled=false

  run_single     fedavg_single_async     configs/rawdata2_patchtst.yaml     federated     --override experiment.mode=federated     --override federated.algorithm=fedavg     --override attack.async_enabled=true     --override attack.async_workers=1     --override attack.async_device=${RUNTIME_DEVICE}

  run_grpc     fedavg_grpc_sync     configs/rawdata2_patchtst.yaml     "${BASE_PORT}"     --override experiment.mode=federated     --override federated.algorithm=fedavg     --override attack.async_enabled=false

  run_grpc     fedavg_grpc_async     configs/rawdata2_patchtst.yaml     "$((BASE_PORT + 1))"     --override experiment.mode=federated     --override federated.algorithm=fedavg     --override attack.async_enabled=true     --override attack.async_workers=1     --override attack.async_device=${RUNTIME_DEVICE}

  run_single     qint8_single_sync     configs/rawdata2_secure_quantized_fedavg.yaml     federated     --override experiment.mode=federated     --override federated.quantization_dtype=int8     --override attack.async_enabled=false

  run_single     qint8_single_async     configs/rawdata2_secure_quantized_fedavg.yaml     federated     --override experiment.mode=federated     --override federated.quantization_dtype=int8     --override attack.async_enabled=true     --override attack.async_workers=1     --override attack.async_device=${RUNTIME_DEVICE}

  run_grpc     qint8_grpc_sync     configs/rawdata2_secure_quantized_fedavg.yaml     "$((BASE_PORT + 2))"     --override experiment.mode=federated     --override federated.quantization_dtype=int8     --override attack.async_enabled=false

  run_grpc     qint8_grpc_async     configs/rawdata2_secure_quantized_fedavg.yaml     "$((BASE_PORT + 3))"     --override experiment.mode=federated     --override federated.quantization_dtype=int8     --override attack.async_enabled=true     --override attack.async_workers=1     --override attack.async_device=${RUNTIME_DEVICE}

  log "all runs finished"
}

main "$@"
