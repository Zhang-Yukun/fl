#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
BASE_OUTPUT="${BASE_OUTPUT:-outputs/ega_targeted_noattack_alt_mse_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-ega-targeted-noattack-mse-alt-v1}"
GROUP_NAME="${GROUP_NAME:-$(basename "${BASE_OUTPUT}")}"
ROUNDS="${ROUNDS:-300}"
PATIENCE="${PATIENCE:-50}"
TRAIN_LR="${TRAIN_LR:-0.001}"

mkdir -p "${BASE_OUTPUT}"

run_case() {
  local run_name="$1"
  local tracking_name="$2"
  shift 2
  local outdir="${BASE_OUTPUT}/${run_name}"
  echo "[$(date '+%F %T')] start ${run_name} device=${RUNTIME_DEVICE}"
  PYTHONPATH=. ${PYTHON_BIN} -m fedlab.entrypoints.train \
    --config configs/ega.yaml \
    --mode federated \
    --override "experiment.output_dir=${outdir}" \
    --override "experiment.name=${run_name}" \
    --override "runtime.device=${RUNTIME_DEVICE}" \
    --override "runtime.seed=2026" \
    --override "runtime.deterministic=true" \
    --override "runtime.num_threads=1" \
    --override "runtime.num_interop_threads=1" \
    --override "tracking.enabled=true" \
    --override "tracking.offline=true" \
    --override "tracking.project=${PROJECT_NAME}" \
    --override "tracking.group=${GROUP_NAME}" \
    --override "tracking.name=${tracking_name}" \
    --override "attack.enabled=false" \
    --override "federated.algorithm=ega_fedavg" \
    --override "federated.rounds=${ROUNDS}" \
    --override "training.epochs=1" \
    --override "training.optimizer=adam" \
    --override "training.lr=${TRAIN_LR}" \
    --override "training.loss=mse" \
    --override "training.patience=${PATIENCE}" \
    --override "training.min_delta=0.0" \
    --override "data.shuffle_train=true" \
    --override "model.dropout=0.1" \
    --override "ega.artifact_path=artifacts/ega/pretrained_codec.pt" \
                    --override "ega.encoded_dtype=int8" \
    --override "ega.encoded_stochastic_rounding=false" \
    --override "ega.encoded_noise_std=0.0" \
    --override "ega.error_feedback=true" \
    "$@"
  echo "[$(date '+%F %T')] finish ${run_name}"
}

run_case \
  "ega_mse_ed152_hd1536_rb3_q127_ema098_pt180" \
  "ega-mse-ed152-hd1536-rb3-q127-ema098-pt180" \
  --override ega.encoded_dim=152 \
  --override ega.hidden_dim=1536 \
  --override ega.residual_blocks=3 \
  --override ega.quantization_level=127 \
  --override ega.normalization_strategy=ema_reported_client_max_abs \
  --override ega.normalization_ema=0.98 \
  --override ega.pretrain.epochs=180 \
  --override ega.pretrain.patience=36 \
  --override ega.pretrain.lr=0.00025 \
  --override ega.pretrain.train_groups=40000 \
  --override ega.pretrain.val_groups=20000 \
  --override ega.pretrain.device=${RUNTIME_DEVICE}

run_case \
  "ega_mse_ed168_hd1536_rb3_q127_ema098_pt180" \
  "ega-mse-ed168-hd1536-rb3-q127-ema098-pt180" \
  --override ega.encoded_dim=168 \
  --override ega.hidden_dim=1536 \
  --override ega.residual_blocks=3 \
  --override ega.quantization_level=127 \
  --override ega.normalization_strategy=ema_reported_client_max_abs \
  --override ega.normalization_ema=0.98 \
  --override ega.pretrain.epochs=180 \
  --override ega.pretrain.patience=36 \
  --override ega.pretrain.lr=0.00025 \
  --override ega.pretrain.train_groups=40000 \
  --override ega.pretrain.val_groups=20000 \
  --override ega.pretrain.device=${RUNTIME_DEVICE}

run_case \
  "ega_mse_ed160_hd2048_rb4_q127_ema097_pt220" \
  "ega-mse-ed160-hd2048-rb4-q127-ema097-pt220" \
  --override ega.encoded_dim=160 \
  --override ega.hidden_dim=2048 \
  --override ega.residual_blocks=4 \
  --override ega.quantization_level=127 \
  --override ega.normalization_strategy=ema_reported_client_max_abs \
  --override ega.normalization_ema=0.97 \
  --override ega.pretrain.epochs=220 \
  --override ega.pretrain.patience=44 \
  --override ega.pretrain.lr=0.0002 \
  --override ega.pretrain.train_groups=50000 \
  --override ega.pretrain.val_groups=25000 \
  --override ega.pretrain.device=${RUNTIME_DEVICE}

run_case \
  "ega_mse_ed152_hd2048_rb3_q159_ema097_pt220" \
  "ega-mse-ed152-hd2048-rb3-q159-ema097-pt220" \
  --override ega.encoded_dim=152 \
  --override ega.hidden_dim=2048 \
  --override ega.residual_blocks=3 \
  --override ega.quantization_level=159 \
  --override ega.normalization_strategy=ema_reported_client_max_abs \
  --override ega.normalization_ema=0.97 \
  --override ega.pretrain.epochs=220 \
  --override ega.pretrain.patience=44 \
  --override ega.pretrain.lr=0.0002 \
  --override ega.pretrain.train_groups=50000 \
  --override ega.pretrain.val_groups=25000 \
  --override ega.pretrain.device=${RUNTIME_DEVICE}

run_case \
  "ega_mse_ed168_hd2048_rb4_q159_ema098_pt220" \
  "ega-mse-ed168-hd2048-rb4-q159-ema098-pt220" \
  --override ega.encoded_dim=168 \
  --override ega.hidden_dim=2048 \
  --override ega.residual_blocks=4 \
  --override ega.quantization_level=159 \
  --override ega.normalization_strategy=ema_reported_client_max_abs \
  --override ega.normalization_ema=0.98 \
  --override ega.pretrain.epochs=220 \
  --override ega.pretrain.patience=44 \
  --override ega.pretrain.lr=0.0002 \
  --override ega.pretrain.train_groups=50000 \
  --override ega.pretrain.val_groups=25000 \
  --override ega.pretrain.device=${RUNTIME_DEVICE}

run_case \
  "ega_mse_ed176_hd1536_rb4_q191_ema097_pt180" \
  "ega-mse-ed176-hd1536-rb4-q191-ema097-pt180" \
  --override ega.encoded_dim=176 \
  --override ega.hidden_dim=1536 \
  --override ega.residual_blocks=4 \
  --override ega.quantization_level=191 \
  --override ega.normalization_strategy=ema_reported_client_max_abs \
  --override ega.normalization_ema=0.97 \
  --override ega.pretrain.epochs=180 \
  --override ega.pretrain.patience=36 \
  --override ega.pretrain.lr=0.00025 \
  --override ega.pretrain.train_groups=40000 \
  --override ega.pretrain.val_groups=20000 \
  --override ega.pretrain.device=${RUNTIME_DEVICE}
