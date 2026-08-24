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
TOPK_FRACTION="${TOPK_FRACTION:-}"
QSGD_LEVELS="${QSGD_LEVELS:-}"
RANDOMK_FRACTION="${RANDOMK_FRACTION:-}"
RANDOMK_SEED="${RANDOMK_SEED:-}"
QINT8_DTYPE="${QINT8_DTYPE:-}"
QINT8_STOCHASTIC_ROUNDING="${QINT8_STOCHASTIC_ROUNDING:-}"
QINT8_SEED="${QINT8_SEED:-}"
QINT8_NOISE_MULTIPLIER="${QINT8_NOISE_MULTIPLIER:-}"
EGA_TRACKING_LABEL="${EGA_TRACKING_LABEL:-ega}"
EGA_ARTIFACT_PATH="${EGA_ARTIFACT_PATH:-}"
EGA_ENCODED_DIM="${EGA_ENCODED_DIM:-}"
EGA_HIDDEN_DIM="${EGA_HIDDEN_DIM:-}"
EGA_RESIDUAL_BLOCKS="${EGA_RESIDUAL_BLOCKS:-}"
EGA_QUANTIZATION_LEVEL="${EGA_QUANTIZATION_LEVEL:-}"
EGA_NORMALIZATION="${EGA_NORMALIZATION:-}"
EGA_INITIAL_NORMALIZATION="${EGA_INITIAL_NORMALIZATION:-}"
EGA_MIN_NORMALIZATION="${EGA_MIN_NORMALIZATION:-}"
EGA_NORMALIZATION_STRATEGY="${EGA_NORMALIZATION_STRATEGY:-}"
EGA_NORMALIZATION_EMA="${EGA_NORMALIZATION_EMA:-}"
EGA_ENCODED_DTYPE="${EGA_ENCODED_DTYPE:-}"
EGA_ENCODED_STOCHASTIC_ROUNDING="${EGA_ENCODED_STOCHASTIC_ROUNDING:-}"
EGA_ENCODED_NOISE_STD="${EGA_ENCODED_NOISE_STD:-}"
EGA_DOWNLOAD_METHOD="${EGA_DOWNLOAD_METHOD:-}"
EGA_DOWNLOAD_DTYPE="${EGA_DOWNLOAD_DTYPE:-}"
EGA_DOWNLOAD_ENCODED_DTYPE="${EGA_DOWNLOAD_ENCODED_DTYPE:-}"
EGA_DOWNLOAD_STOCHASTIC_ROUNDING="${EGA_DOWNLOAD_STOCHASTIC_ROUNDING:-}"
EGA_DOWNLOAD_TRAINABLE_ONLY="${EGA_DOWNLOAD_TRAINABLE_ONLY:-}"
EGA_ERROR_FEEDBACK="${EGA_ERROR_FEEDBACK:-}"
EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE:-}"
EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS:-}"
EGA_PRETRAIN_PATIENCE="${EGA_PRETRAIN_PATIENCE:-}"
EGA_PRETRAIN_MIN_DELTA="${EGA_PRETRAIN_MIN_DELTA:-}"
EGA_PRETRAIN_BATCH_SIZE="${EGA_PRETRAIN_BATCH_SIZE:-}"
EGA_PRETRAIN_LR="${EGA_PRETRAIN_LR:-}"
EGA_PRETRAIN_TRAIN_GROUPS="${EGA_PRETRAIN_TRAIN_GROUPS:-}"
EGA_PRETRAIN_VAL_GROUPS="${EGA_PRETRAIN_VAL_GROUPS:-}"
EGA_PRETRAIN_SEED="${EGA_PRETRAIN_SEED:-}"

emit_optional_override() {
  local key="$1"
  local value="${2:-}"
  if [[ -n "${value}" ]]; then
    printf -- '--override
%s=%s
' "${key}" "${value}"
  fi
}

run_train() {
  local mode="$1"
  local config="$2"
  local run_name="$3"
  local short_name="$4"
  shift 4
  local outdir="${BASE_OUTPUT}/${run_name}"
  echo "[$(date '+%F %T')] start ${run_name} on gpu ${GPU_ID}"
  PYTHONPATH=. ${PYTHON_BIN} -m fedlab.entrypoints.train \
    --config "${config}" \
    --mode "${mode}" \
    --override "experiment.output_dir=${outdir}" \
    --override "experiment.mode=${mode}" \
    --override "runtime.device=${RUNTIME_DEVICE}" \
    --override "runtime.seed=2026" \
    --override "runtime.deterministic=true" \
    --override "runtime.num_threads=1" \
    --override "runtime.num_interop_threads=1" \
    --override "tracking.enabled=true" \
    --override "tracking.offline=true" \
    --override "tracking.project=${PROJECT_NAME}" \
    --override "tracking.group=$(basename "${BASE_OUTPUT}")" \
    --override "tracking.name=${short_name}" \
    --override "training.patience=${PATIENCE}" \
    --override "training.min_delta=0.0" \
    "$@"
  echo "[$(date '+%F %T')] finish ${run_name}"
}

if [[ "${WORKER_KIND}" == "gpu0" ]]; then
  run_train centralized configs/rawdata2_centralized.yaml centralized_500r_pat50 cen-500r-pat50 \
    --override "centralized.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_fedavg.yaml fedavg_single_sync_500r_pat50 fedavg-single-sync-500r-pat50 \
    --override "federated.algorithm=fedavg" \
    --override "federated.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_fedlab_topk.yaml topk_single_sync_500r_pat50 topk-single-sync-500r-pat50 \
    --override "federated.algorithm=sparse_fedavg" \
    --override "federated.rounds=${ROUNDS}" \
    $(emit_optional_override 'federated.topk_fraction' "${TOPK_FRACTION}")

  run_train federated configs/rawdata2_qsgd.yaml qsgd_single_sync_500r_pat50 qsgd-single-sync-500r-pat50 \
    --override "federated.algorithm=qsgd_fedavg" \
    --override "federated.rounds=${ROUNDS}" \
    $(emit_optional_override 'federated.qsgd_levels' "${QSGD_LEVELS}")

  run_train federated configs/rawdata2_randomk.yaml randomk_single_sync_500r_pat50 randomk-single-sync-500r-pat50 \
    --override "federated.algorithm=randomk_fedavg" \
    --override "federated.rounds=${ROUNDS}" \
    $(emit_optional_override 'federated.topk_fraction' "${RANDOMK_FRACTION}") \
    $(emit_optional_override 'federated.randomk_seed' "${RANDOMK_SEED}")
else
  run_train federated configs/rawdata2_sign.yaml sign_single_sync_500r_pat50 sign-single-sync-500r-pat50 \
    --override "federated.algorithm=sign_fedavg" \
    --override "federated.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_adaptive_clipped_rdp_fedavg.yaml adaptive_single_sync_500r_pat50 adaptive-single-sync-500r-pat50 \
    --override "federated.algorithm=adaptive_clipped_rdp_fedavg" \
    --override "federated.rounds=${ROUNDS}"

  run_train federated configs/rawdata2_secure_quantized_fedavg.yaml qint8_single_sync_500r_pat50 secure-quantized-single-sync-500r-pat50 \
    --override "federated.algorithm=secure_quantized_fedavg" \
    --override "federated.rounds=${ROUNDS}" \
    $(emit_optional_override 'federated.quantization_dtype' "${QINT8_DTYPE}") \
    $(emit_optional_override 'federated.quantization_stochastic_rounding' "${QINT8_STOCHASTIC_ROUNDING}") \
    $(emit_optional_override 'federated.quantization_seed' "${QINT8_SEED}") \
    $(emit_optional_override 'privacy.noise_multiplier' "${QINT8_NOISE_MULTIPLIER}")

  run_train federated configs/rawdata2_ega.yaml ega_single_sync_500r_pat50 ${EGA_TRACKING_LABEL}-single-sync-500r-pat50 \
    --override "federated.algorithm=ega_fedavg" \
    --override "federated.rounds=${ROUNDS}" \
    $(emit_optional_override 'ega.artifact_path' "${EGA_ARTIFACT_PATH}") \
    $(emit_optional_override 'ega.encoded_dim' "${EGA_ENCODED_DIM}") \
    $(emit_optional_override 'ega.hidden_dim' "${EGA_HIDDEN_DIM}") \
    $(emit_optional_override 'ega.residual_blocks' "${EGA_RESIDUAL_BLOCKS}") \
    $(emit_optional_override 'ega.quantization_level' "${EGA_QUANTIZATION_LEVEL}") \
    $(emit_optional_override 'ega.normalization' "${EGA_NORMALIZATION}") \
    $(emit_optional_override 'ega.initial_normalization' "${EGA_INITIAL_NORMALIZATION}") \
    $(emit_optional_override 'ega.min_normalization' "${EGA_MIN_NORMALIZATION}") \
    $(emit_optional_override 'ega.normalization_strategy' "${EGA_NORMALIZATION_STRATEGY}") \
    $(emit_optional_override 'ega.normalization_ema' "${EGA_NORMALIZATION_EMA}") \
    $(emit_optional_override 'ega.encoded_dtype' "${EGA_ENCODED_DTYPE}") \
    $(emit_optional_override 'ega.encoded_stochastic_rounding' "${EGA_ENCODED_STOCHASTIC_ROUNDING}") \
    $(emit_optional_override 'ega.encoded_noise_std' "${EGA_ENCODED_NOISE_STD}") \
    $(emit_optional_override 'ega.download_method' "${EGA_DOWNLOAD_METHOD}") \
    $(emit_optional_override 'ega.download_dtype' "${EGA_DOWNLOAD_DTYPE}") \
    $(emit_optional_override 'ega.download_encoded_dtype' "${EGA_DOWNLOAD_ENCODED_DTYPE}") \
    $(emit_optional_override 'ega.download_stochastic_rounding' "${EGA_DOWNLOAD_STOCHASTIC_ROUNDING}") \
    $(emit_optional_override 'ega.download_trainable_only' "${EGA_DOWNLOAD_TRAINABLE_ONLY}") \
    $(emit_optional_override 'ega.error_feedback' "${EGA_ERROR_FEEDBACK}") \
    $(emit_optional_override 'ega.pretrain.device' "${EGA_PRETRAIN_DEVICE}") \
    $(emit_optional_override 'ega.pretrain.epochs' "${EGA_PRETRAIN_EPOCHS}") \
    $(emit_optional_override 'ega.pretrain.patience' "${EGA_PRETRAIN_PATIENCE}") \
    $(emit_optional_override 'ega.pretrain.min_delta' "${EGA_PRETRAIN_MIN_DELTA}") \
    $(emit_optional_override 'ega.pretrain.batch_size' "${EGA_PRETRAIN_BATCH_SIZE}") \
    $(emit_optional_override 'ega.pretrain.lr' "${EGA_PRETRAIN_LR}") \
    $(emit_optional_override 'ega.pretrain.train_groups' "${EGA_PRETRAIN_TRAIN_GROUPS}") \
    $(emit_optional_override 'ega.pretrain.val_groups' "${EGA_PRETRAIN_VAL_GROUPS}") \
    $(emit_optional_override 'ega.pretrain.seed' "${EGA_PRETRAIN_SEED}")
fi
