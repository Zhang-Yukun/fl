#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:?RUNTIME_DEVICE is required}"
BASE_OUTPUT="${BASE_OUTPUT:?BASE_OUTPUT is required}"
LOSS_NAME="${LOSS_NAME:?LOSS_NAME is required}"
PROJECT_NAME="${PROJECT_NAME:?PROJECT_NAME is required}"
GROUP_NAME="${GROUP_NAME:-$(basename "${BASE_OUTPUT}")}"
ROUNDS="${ROUNDS:-300}"
PATIENCE="${PATIENCE:-50}"
TRAIN_LR="${TRAIN_LR:-0.001}"

run_case() {
  local run_name="$1"
  local tracking_name="$2"
  shift 2
  local outdir="${BASE_OUTPUT}/${run_name}"
  echo "[$(date '+%F %T')] start ${run_name} loss=${LOSS_NAME} device=${RUNTIME_DEVICE}"
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
    --override "evaluation.mode=protocol" \
    --override "federated.algorithm=ega_fedavg" \
    --override "federated.rounds=${ROUNDS}" \
    --override "federated.local_epochs=1" \
    --override "training.optimizer=adam" \
    --override "training.lr=${TRAIN_LR}" \
    --override "training.loss=${LOSS_NAME}" \
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

run_suite_mse() {
  run_case \
    "ega_mse_ed144_hd1536_rb3_q127_ema095_pt150" \
    "ega-mse-ed144-hd1536-rb3-q127-ema095-pt150" \
    --override ega.encoded_dim=144 \
    --override ega.hidden_dim=1536 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=127 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mse_ed160_hd1536_rb3_q127_ema095_pt150" \
    "ega-mse-ed160-hd1536-rb3-q127-ema095-pt150" \
    --override ega.encoded_dim=160 \
    --override ega.hidden_dim=1536 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=127 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mse_ed160_hd1024_rb2_q159_ema095_pt150" \
    "ega-mse-ed160-hd1024-rb2-q159-ema095-pt150" \
    --override ega.encoded_dim=160 \
    --override ega.hidden_dim=1024 \
    --override ega.residual_blocks=2 \
    --override ega.quantization_level=159 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mse_ed176_hd1536_rb3_q159_ema095_pt150" \
    "ega-mse-ed176-hd1536-rb3-q159-ema095-pt150" \
    --override ega.encoded_dim=176 \
    --override ega.hidden_dim=1536 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=159 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mse_ed144_hd2048_rb3_q159_ema095_pt200" \
    "ega-mse-ed144-hd2048-rb3-q159-ema095-pt200" \
    --override ega.encoded_dim=144 \
    --override ega.hidden_dim=2048 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=159 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=200 \
    --override ega.pretrain.patience=40 \
    --override ega.pretrain.lr=0.0002 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mse_ed160_hd1536_rb4_q127_ema097_pt150" \
    "ega-mse-ed160-hd1536-rb4-q127-ema097-pt150" \
    --override ega.encoded_dim=160 \
    --override ega.hidden_dim=1536 \
    --override ega.residual_blocks=4 \
    --override ega.quantization_level=127 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.97 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}
}

run_suite_mae() {
  run_case \
    "ega_mae_ed160_hd1536_rb3_q127_ema095_pt150" \
    "ega-mae-ed160-hd1536-rb3-q127-ema095-pt150" \
    --override ega.encoded_dim=160 \
    --override ega.hidden_dim=1536 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=127 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mae_ed176_hd1536_rb3_q127_ema095_pt150" \
    "ega-mae-ed176-hd1536-rb3-q127-ema095-pt150" \
    --override ega.encoded_dim=176 \
    --override ega.hidden_dim=1536 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=127 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mae_ed160_hd2048_rb3_q159_ema095_pt200" \
    "ega-mae-ed160-hd2048-rb3-q159-ema095-pt200" \
    --override ega.encoded_dim=160 \
    --override ega.hidden_dim=2048 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=159 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=200 \
    --override ega.pretrain.patience=40 \
    --override ega.pretrain.lr=0.0002 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mae_ed176_hd2048_rb3_q159_ema097_pt200" \
    "ega-mae-ed176-hd2048-rb3-q159-ema097-pt200" \
    --override ega.encoded_dim=176 \
    --override ega.hidden_dim=2048 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=159 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.97 \
    --override ega.pretrain.epochs=200 \
    --override ega.pretrain.patience=40 \
    --override ega.pretrain.lr=0.0002 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mae_ed144_hd1536_rb3_q159_ema095_pt150" \
    "ega-mae-ed144-hd1536-rb3-q159-ema095-pt150" \
    --override ega.encoded_dim=144 \
    --override ega.hidden_dim=1536 \
    --override ega.residual_blocks=3 \
    --override ega.quantization_level=159 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.95 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case \
    "ega_mae_ed160_hd1536_rb4_q127_ema097_pt150" \
    "ega-mae-ed160-hd1536-rb4-q127-ema097-pt150" \
    --override ega.encoded_dim=160 \
    --override ega.hidden_dim=1536 \
    --override ega.residual_blocks=4 \
    --override ega.quantization_level=127 \
    --override ega.normalization_strategy=ema_reported_client_max_abs \
    --override ega.normalization_ema=0.97 \
    --override ega.pretrain.epochs=150 \
    --override ega.pretrain.patience=30 \
    --override ega.pretrain.lr=0.0003 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}
}

case "${LOSS_NAME}" in
  mse)
    run_suite_mse
    ;;
  mae)
    run_suite_mae
    ;;
  *)
    echo "Unsupported LOSS_NAME=${LOSS_NAME}" >&2
    exit 1
    ;;
esac
