#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-conda run -n torch_env python}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:?RUNTIME_DEVICE is required}"
BASE_OUTPUT="${BASE_OUTPUT:?BASE_OUTPUT is required}"
PROJECT_NAME="${PROJECT_NAME:?PROJECT_NAME is required}"
WORKER_KIND="${WORKER_KIND:?WORKER_KIND is required}"
ROUNDS="${ROUNDS:-500}"
PATIENCE="${PATIENCE:-50}"

run_train() {
  local mode="$1"
  local config="$2"
  local run_name="$3"
  local short_name="$4"
  shift 4
  local outdir="${BASE_OUTPUT}/${run_name}"
  echo "[$(date '+%F %T')] start ${run_name} on gpu ${GPU_ID}"
  PYTHONPATH=. ${PYTHON_BIN} -m fedlab.entrypoints.train     --config "${config}"     --mode "${mode}"     --override "experiment.output_dir=${outdir}"     --override "experiment.mode=${mode}"     --override "runtime.device=${RUNTIME_DEVICE}"     --override "runtime.seed=2026"     --override "runtime.deterministic=true"     --override "runtime.num_threads=1"     --override "runtime.num_interop_threads=1"     --override "tracking.enabled=true"     --override "tracking.offline=true"     --override "tracking.project=${PROJECT_NAME}"     --override "tracking.group=$(basename "${BASE_OUTPUT}")"     --override "tracking.name=${short_name}"     --override "training.patience=${PATIENCE}"     --override "training.min_delta=0.0"     "$@"
  echo "[$(date '+%F %T')] finish ${run_name}"
}

if [[ "${WORKER_KIND}" == "gpu0" ]]; then
  run_train centralized configs/rawdata2_centralized.yaml centralized_500r_pat50 cen-500r-pat50     --override "centralized.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_fedavg.yaml fedavg_single_sync_500r_pat50 fedavg-single-sync-500r-pat50     --override "federated.algorithm=fedavg"     --override "federated.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_fedlab_topk.yaml topk_single_sync_500r_pat50 topk-single-sync-500r-pat50     --override "federated.algorithm=sparse_fedavg"     --override "federated.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_qsgd.yaml qsgd_single_sync_500r_pat50 qsgd-single-sync-500r-pat50     --override "federated.algorithm=qsgd_fedavg"     --override "federated.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_randomk.yaml randomk_single_sync_500r_pat50 randomk-single-sync-500r-pat50     --override "federated.algorithm=randomk_fedavg"     --override "federated.rounds=${ROUNDS}"
else
  run_train federated configs/rawdata2_sign.yaml sign_single_sync_500r_pat50 sign-single-sync-500r-pat50     --override "federated.algorithm=sign_fedavg"     --override "federated.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_adaptive_clipped_rdp_fedavg.yaml adaptive_single_sync_500r_pat50 adaptive-single-sync-500r-pat50     --override "federated.algorithm=adaptive_clipped_rdp_fedavg"     --override "federated.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_secure_quantized_fedavg.yaml qint8_single_sync_500r_pat50 qint8-single-sync-500r-pat50     --override "federated.algorithm=secure_quantized_fedavg"     --override "federated.rounds=${ROUNDS}"     --override "federated.quantization_dtype=int8"     --override "federated.quantization_stochastic_rounding=false"

  run_train federated configs/rawdata2_ega.yaml ega_ed192_single_sync_500r_pat50 ega-ed192-single-sync-500r-pat50     --override "federated.algorithm=ega_fedavg"     --override "federated.rounds=${ROUNDS}"     --override "ega.artifact_path=artifacts/ega/rawdata2_ega_h240_v1.pt"     --override "ega.encoded_dim=192"     --override "ega.hidden_dim=1024"     --override "ega.residual_blocks=2"     --override "ega.quantization_level=127"     --override "ega.normalization=0.00025"     --override "ega.initial_normalization=0.00025"     --override "ega.min_normalization=0.000001"     --override "ega.normalization_strategy=ema_reported_client_max_abs"     --override "ega.normalization_ema=0.9"     --override "ega.encoded_dtype=int8"     --override "ega.encoded_stochastic_rounding=false"     --override "ega.encoded_noise_std=0.0"     --override "ega.download_dtype=int8"     --override "ega.download_method=dense"     --override "ega.download_stochastic_rounding=false"     --override "ega.download_trainable_only=true"     --override "ega.download_encoded_dtype=int8"     --override "ega.error_feedback=true"     --override "ega.pretrain.device=${RUNTIME_DEVICE}"     --override "ega.pretrain.epochs=100"     --override "ega.pretrain.patience=20"     --override "ega.pretrain.min_delta=0.0"     --override "ega.pretrain.batch_size=128"     --override "ega.pretrain.lr=0.0005"     --override "ega.pretrain.train_groups=30000"     --override "ega.pretrain.val_groups=15000"     --override "ega.pretrain.seed=2026"
fi
