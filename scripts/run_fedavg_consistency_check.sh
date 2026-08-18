#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CONFIG="${CONFIG:-configs/rawdata2_fedavg_small_deterministic.yaml}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/fedavg_consistency}"
PORT_BASE="${PORT_BASE:-55100}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_single() {
  local name="$1"
  local async_flag="$2"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.train \
    --config "${CONFIG}" \
    --mode federated \
    --override "experiment.output_dir=${BASE_OUTPUT}/${name}" \
    --override "runtime.device=cuda:0" \
    --override "attack.async_enabled=${async_flag}" \
    --override "attack.device=cuda:0"
}

run_grpc() {
  local name="$1"
  local async_flag="$2"
  local port="$3"
  local address="127.0.0.1:${port}"
  local outdir="${BASE_OUTPUT}/${name}"
  mkdir -p "${outdir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.server \
    --config "${CONFIG}" \
    --override "experiment.output_dir=${outdir}" \
    --override "runtime.device=cuda:0" \
    --override "attack.async_enabled=${async_flag}" \
    --override "attack.device=cuda:0" \
    --override "grpc.address=${address}" \
    --override "grpc.server_address=${address}" \
    --override "grpc.poll_seconds=0.2" \
    > "${outdir}/server.log" 2>&1 &
  local server_pid=$!
  sleep 3
  local client_pids=()
  for client_id in Nd2O3 CeO2 La2O3; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.client \
      --client-id "${client_id}" \
      --config "${CONFIG}" \
      --override "experiment.output_dir=${outdir}" \
      --override "runtime.device=cuda:0" \
      --override "attack.async_enabled=${async_flag}" \
      --override "attack.device=cuda:0" \
      --override "grpc.address=${address}" \
      --override "grpc.server_address=${address}" \
      --override "grpc.poll_seconds=0.2" \
      > "${outdir}/client_${client_id}.log" 2>&1 &
    client_pids+=("$!")
  done
  for pid in "${client_pids[@]}"; do
    wait "${pid}"
  done
  wait "${server_pid}"
}

mkdir -p "${BASE_OUTPUT}"
run_single single_sync false
run_single single_async true
PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.tools.compare_fedavg_consistency "${BASE_OUTPUT}/single_sync" "${BASE_OUTPUT}/single_async"

run_grpc grpc_sync false "${PORT_BASE}"
run_grpc grpc_async true "$((PORT_BASE + 1))"
PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.tools.compare_fedavg_consistency --ignore-transport "${BASE_OUTPUT}/grpc_sync" "${BASE_OUTPUT}/grpc_async"
