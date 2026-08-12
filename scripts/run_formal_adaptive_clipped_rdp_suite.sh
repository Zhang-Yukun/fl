#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
BASE_PORT="${BASE_PORT:-56100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/formal_adaptive_clipped_rdp_suite_$(date +%Y%m%d_%H%M%S)}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"
POLL_SECONDS="${POLL_SECONDS:-1.0}"

COMMON_OVERRIDES=(
  --override experiment.mode=federated
  --override tracking.enabled=true
  --override tracking.offline=true
  --override runtime.seed=2026
  --override runtime.deterministic=true
  --override runtime.num_threads=1
  --override runtime.num_interop_threads=1
  --override data.shuffle_train=false
  --override model.dropout=0.0
  --override attack.seed=2026
  --override adaptive_clipped_rdp.seed=2026
)

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
  shift 1
  local outdir="${OUTPUT_ROOT}/${name}"

  log "starting ${name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.train \
    --config configs/rawdata2_adaptive_clipped_rdp_fedavg_deterministic.yaml \
    --mode federated \
    --override "experiment.output_dir=${outdir}" \
    --override "runtime.device=${RUNTIME_DEVICE}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
  log "finished ${name}"
}

run_grpc() {
  local name="$1"
  local port="$2"
  shift 2

  local outdir="${OUTPUT_ROOT}/${name}"
  local address="127.0.0.1:${port}"
  local server_pid=""
  local client_pids=()

  mkdir -p "${outdir}"
  log "starting ${name} on ${address}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.server \
    --config configs/rawdata2_adaptive_clipped_rdp_fedavg_deterministic.yaml \
    --override "experiment.output_dir=${outdir}" \
    --override "runtime.device=${RUNTIME_DEVICE}" \
    --override "grpc.address=${address}" \
    --override "grpc.server_address=${address}" \
    --override "grpc.poll_seconds=${POLL_SECONDS}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@" \
    > "${outdir}/server.log" 2>&1 &
  server_pid=$!

  sleep "${STARTUP_WAIT_SECONDS}"

  for client_id in Nd2O3 CeO2 La2O3; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.client \
      --client-id "${client_id}" \
      --config configs/rawdata2_adaptive_clipped_rdp_fedavg_deterministic.yaml \
      --override "experiment.output_dir=${outdir}" \
      --override "runtime.device=${RUNTIME_DEVICE}" \
      --override "grpc.address=${address}" \
      --override "grpc.server_address=${address}" \
      --override "grpc.poll_seconds=${POLL_SECONDS}" \
      "${COMMON_OVERRIDES[@]}" \
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
  log "adaptive clipped RDP suite output root: ${OUTPUT_ROOT}"
  mkdir -p "${OUTPUT_ROOT}"

  run_single adaptive_clipped_rdp_single_sync \
    --override attack.async_enabled=false

  run_single adaptive_clipped_rdp_single_async \
    --override attack.async_enabled=true \
    --override attack.async_workers=1 \
    --override "attack.device=${RUNTIME_DEVICE}"

  run_grpc adaptive_clipped_rdp_grpc_sync "${BASE_PORT}" \
    --override attack.async_enabled=false

  run_grpc adaptive_clipped_rdp_grpc_async "$((BASE_PORT + 1))" \
    --override attack.async_enabled=true \
    --override attack.async_workers=1 \
    --override "attack.device=${RUNTIME_DEVICE}"

  log "all runs finished"
}

main "$@"
