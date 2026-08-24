#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/repro_pat50/secure_qfloat16_bidir_seed2026_pat50_payloadv1}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m fedlab.entrypoints.train \
  --config configs/secure_quantized_fedavg.yaml \
  --mode federated \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override runtime.device=cuda:0 \
  --override runtime.seed=2026 \
  --override runtime.deterministic=true \
  --override training.patience=50 \
  --override tracking.enabled=true \
  --override tracking.offline=true \
  --override attack.seed=2026 \
  --override federated.quantization_dtype=float16 \
  --override federated.quantization_stochastic_rounding=false \
  --override federated.quantization_seed=2026 \
  "$@"
