#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/oracle_attackfreq5_1000r_pat100_9algo_4mode_mae_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-fl-oracle-attackfreq5-1000r-pat100-v1}"
BASE_PORT="${BASE_PORT:-58000}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-5}"
POLL_SECONDS="${POLL_SECONDS:-1.0}"
RUN_CENTRALIZED="${RUN_CENTRALIZED:-true}"
ROUNDS="${ROUNDS:-1000}"
PATIENCE="${PATIENCE:-100}"
LOSS_TAG="${LOSS_TAG:-mae}"

SELECT_MODES="${SELECT_MODES:-}"

usage() {
  cat <<'EOF'
Usage: bash SCRIPT [--modes centralized,single_sync,single_async,grpc_sync,grpc_async]

Examples:
  bash SCRIPT --modes single_sync
  bash SCRIPT --modes single_sync,grpc_sync
  SELECT_MODES=single_async,grpc_async bash SCRIPT
EOF
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

tracking_args() {
  local tracking_name="$1"
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
runtime.seed=2026
--override
runtime.deterministic=true
--override
runtime.num_threads=1
--override
runtime.num_interop_threads=1
--override
data.shuffle_train=false
--override
model.dropout=0.0
' "${RUNTIME_DEVICE}"
}

centralized_round_args() {
  if [[ -n "${ROUNDS}" ]]; then
    printf -- '--override
centralized.rounds=%s
' "${ROUNDS}"
  fi
  if [[ -n "${PATIENCE}" ]]; then
    printf -- '--override
training.patience=%s
--override
training.min_delta=0.0
' "${PATIENCE}"
  fi
  printf -- '--override
training.loss=mae
'
  printf -- '--override
training.lr=0.0005
'
}

federated_common_args() {
  printf -- '--override
experiment.mode=federated
--override
transport.upload_mode=update
--override
transport.download_mode=model
--override
evaluation.mode=oracle_full_update
--override
attack.enabled=true
--override
attack.async_enabled=false
--override
attack.frequency_rounds=5
--override
attack.client_selection=all
--override
attack.clients_per_round=3
' 
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
  printf -- '--override
training.loss=mae
'
  printf -- '--override
training.lr=0.0005
--override
attack.lr=0.001
--override
attack.optimizer=adam
'
}

run_single() {
  local run_name="$1"
  local tracking_name="$2"
  local config="$3"
  shift 3
  local outdir="${BASE_OUTPUT}/${run_name}"
  local -a cmd=(
    "${PYTHON_BIN}" -m fedlab.entrypoints.train
    --config "${config}"
    --mode federated
    --override "experiment.output_dir=${outdir}"
    --override "experiment.name=${run_name}"
  )
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(tracking_args "${tracking_name}")
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(base_runtime_args)
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(federated_common_args)
  cmd+=("$@")

  log "starting ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${cmd[@]}"
  log "finished ${run_name}"
}

run_grpc() {
  local run_name="$1"
  local tracking_name="$2"
  local config="$3"
  local port="$4"
  shift 4
  local outdir="${BASE_OUTPUT}/${run_name}"
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
  done < <(tracking_args "${tracking_name}")
  while IFS= read -r line; do
    server_cmd+=("${line}")
  done < <(base_runtime_args)
  while IFS= read -r line; do
    server_cmd+=("${line}")
  done < <(federated_common_args)
  server_cmd+=("$@")

  log "starting ${run_name} on ${address}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${server_cmd[@]}" > "${outdir}/server.log" 2>&1 &
  server_pid=$!
  sleep "${STARTUP_WAIT_SECONDS}"

  for client_id in Nd2O3 CeO2 La2O3; do
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
    done < <(federated_common_args)
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
  local run_name="$1"
  local tracking_name="$2"
  local config="$3"
  shift 3
  local outdir="${BASE_OUTPUT}/${run_name}"
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
  done < <(tracking_args "${tracking_name}")
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(base_runtime_args)
  while IFS= read -r line; do
    cmd+=("${line}")
  done < <(centralized_round_args)
  cmd+=("$@")

  log "starting ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${cmd[@]}"
  log "finished ${run_name}"
}

main() {
  mkdir -p "${BASE_OUTPUT}"
  log "output root: ${BASE_OUTPUT}"
  log "gpu=${GPU_ID} device=${RUNTIME_DEVICE} project=${PROJECT_NAME}"

  if [[ "${RUN_CENTRALIZED}" == "true" ]] && mode_enabled centralized; then
    run_centralized \
      centralized_uupdate_dmodel_oracle_attackfreq5_1000r_pat100 \
      centralized-oracle-attackfreq5-1000r-pat100 \
      configs/rawdata2_centralized.yaml
  fi

  local -a modes=(single_sync single_async grpc_sync grpc_async)
  local port="${BASE_PORT}"
  local mode
  for mode in "${modes[@]}"; do
    if ! mode_enabled "${mode}"; then
      continue
    fi
    if [[ "${mode}" == grpc_* ]]; then
      run_grpc \
        "fedavg_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "fedavg-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_fedavg.yaml \
        "${port}" \
        --override federated.algorithm=fedavg
      port=$((port + 1))

      run_grpc \
        "topk_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "topk10-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_fedlab_topk.yaml \
        "${port}" \
        --override federated.algorithm=sparse_fedavg \
        --override federated.topk_fraction=0.10
      port=$((port + 1))

      run_grpc \
        "qsgd_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "qsgd63-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_qsgd.yaml \
        "${port}" \
        --override federated.algorithm=qsgd_fedavg \
        --override federated.qsgd_levels=63
      port=$((port + 1))

      run_grpc \
        "randomk_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "randomk10-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_randomk.yaml \
        "${port}" \
        --override federated.algorithm=randomk_fedavg \
        --override federated.topk_fraction=0.10 \
        --override federated.randomk_seed=2026
      port=$((port + 1))

      run_grpc \
        "sign_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "sign-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_sign.yaml \
        "${port}" \
        --override federated.algorithm=sign_fedavg
      port=$((port + 1))

      run_grpc \
        "adaptive_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "adaptive-rdp-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_adaptive_clipped_rdp_fedavg_deterministic.yaml \
        "${port}" \
        --override federated.algorithm=adaptive_clipped_rdp_fedavg \
        --override adaptive_clipped_rdp.seed=2026
      port=$((port + 1))

      run_grpc \
        "qint8_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "qint8-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_secure_quantized_fedavg.yaml \
        "${port}" \
        --override federated.algorithm=secure_quantized_fedavg \
        --override federated.quantization_dtype=int8 \
        --override federated.quantization_stochastic_rounding=false \
        --override federated.quantization_seed=2026 \
        --override privacy.noise_multiplier=0.0
      port=$((port + 1))

      run_grpc \
        "ega_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "ega-ed128-dm-ega-pcq127-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_ega.yaml \
        "${port}" \
        --override federated.algorithm=ega_fedavg \
        --override ega.artifact_path=artifacts/ega/ega_ed128_dm_ega_pc_q127.pt \
        --override ega.encoded_dim=128 \
        --override ega.hidden_dim=1024 \
        --override ega.residual_blocks=2 \
        --override ega.download_method=ega \
        --override ega.download_dtype=float32 \
        --override ega.download_encoded_dtype=int8 \
        --override ega.download_encoded_stochastic_rounding=false \
        --override ega.download_trainable_only=true \
        --override ega.download_quantization_level=127 \
        --override ega.download_min_normalization=1e-6 \
        --override ega.download_predictive_coding=true \
        --override ega.normalization_strategy=ema_reported_client_max_abs \
        --override ega.normalization_ema=0.9 \
        --override ega.quantization_level=127 \
        --override ega.encoded_dtype=int8 \
        --override ega.encoded_stochastic_rounding=false \
        --override ega.error_feedback=true \
        --override ega.pretrain.device=${RUNTIME_DEVICE}
      port=$((port + 1))
    else
      local async_flag=false
      local async_workers=()
      if [[ "${mode}" == "single_async" ]]; then
        async_flag=true
        async_workers=(--override attack.async_workers=1)
      fi

      run_single \
        "fedavg_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "fedavg-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_fedavg.yaml \
        --override federated.algorithm=fedavg \
        --override attack.async_enabled=${async_flag} \
        "${async_workers[@]}"

      run_single \
        "topk_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "topk10-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_fedlab_topk.yaml \
        --override federated.algorithm=sparse_fedavg \
        --override federated.topk_fraction=0.10 \
        --override attack.async_enabled=${async_flag} \
        "${async_workers[@]}"

      run_single \
        "qsgd_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "qsgd63-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_qsgd.yaml \
        --override federated.algorithm=qsgd_fedavg \
        --override federated.qsgd_levels=63 \
        --override attack.async_enabled=${async_flag} \
        "${async_workers[@]}"

      run_single \
        "randomk_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "randomk10-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_randomk.yaml \
        --override federated.algorithm=randomk_fedavg \
        --override federated.topk_fraction=0.10 \
        --override federated.randomk_seed=2026 \
        --override attack.async_enabled=${async_flag} \
        "${async_workers[@]}"

      run_single \
        "sign_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "sign-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_sign.yaml \
        --override federated.algorithm=sign_fedavg \
        --override attack.async_enabled=${async_flag} \
        "${async_workers[@]}"

      run_single \
        "adaptive_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "adaptive-rdp-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_adaptive_clipped_rdp_fedavg_deterministic.yaml \
        --override federated.algorithm=adaptive_clipped_rdp_fedavg \
        --override adaptive_clipped_rdp.seed=2026 \
        --override attack.async_enabled=${async_flag} \
        "${async_workers[@]}"

      run_single \
        "qint8_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "qint8-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_secure_quantized_fedavg.yaml \
        --override federated.algorithm=secure_quantized_fedavg \
        --override federated.quantization_dtype=int8 \
        --override federated.quantization_stochastic_rounding=false \
        --override federated.quantization_seed=2026 \
        --override privacy.noise_multiplier=0.0 \
        --override attack.async_enabled=${async_flag} \
        "${async_workers[@]}"

      run_single \
        "ega_${mode}_uupdate_dmodel_oracle_attackfreq5_1000r_pat100" \
        "ega-ed128-dm-ega-pcq127-${mode}-uupdate-dmodel-oracle-attackfreq5-1000r-pat100" \
        configs/rawdata2_ega.yaml \
        --override federated.algorithm=ega_fedavg \
        --override ega.artifact_path=artifacts/ega/ega_ed128_dm_ega_pc_q127.pt \
        --override ega.encoded_dim=128 \
        --override ega.hidden_dim=1024 \
        --override ega.residual_blocks=2 \
        --override ega.download_method=ega \
        --override ega.download_dtype=float32 \
        --override ega.download_encoded_dtype=int8 \
        --override ega.download_encoded_stochastic_rounding=false \
        --override ega.download_trainable_only=true \
        --override ega.download_quantization_level=127 \
        --override ega.download_min_normalization=1e-6 \
        --override ega.download_predictive_coding=true \
        --override ega.normalization_strategy=ema_reported_client_max_abs \
        --override ega.normalization_ema=0.9 \
        --override ega.quantization_level=127 \
        --override ega.encoded_dtype=int8 \
        --override ega.encoded_stochastic_rounding=false \
        --override ega.error_feedback=true \
        --override ega.pretrain.device=${RUNTIME_DEVICE} \
        --override attack.async_enabled=${async_flag} \
        "${async_workers[@]}"
    fi
  done

  log "suite finished"
}

parse_args "$@"
main
