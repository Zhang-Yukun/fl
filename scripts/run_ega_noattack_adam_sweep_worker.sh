#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:?RUNTIME_DEVICE is required}"
BASE_OUTPUT="${BASE_OUTPUT:?BASE_OUTPUT is required}"
LOSS_NAME="${LOSS_NAME:?LOSS_NAME is required}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-ega-adam-noattack-explore-v1}"
ROUNDS="${ROUNDS:-500}"
PATIENCE="${PATIENCE:-50}"
TRAIN_LR="${TRAIN_LR:-0.001}"
GROUP_NAME="${GROUP_NAME:-$(basename "${BASE_OUTPUT}")}" 

run_case() {
  local run_name="$1"
  local tracking_name="$2"
  shift 2
  local outdir="${BASE_OUTPUT}/${run_name}"
  echo "[$(date '+%F %T')] start ${run_name} on gpu ${GPU_ID} loss=${LOSS_NAME}"
  echo "[$(date '+%F %T')] finish ${run_name}"
}

loss_tag="${LOSS_NAME}"

run_case   "ega_${loss_tag}_ed128_hd1024_rb2_q127"   "ega-${loss_tag}-ed128-hd1024-rb2-q127"   --override ega.encoded_dim=128   --override ega.hidden_dim=1024   --override ega.residual_blocks=2

run_case   "ega_${loss_tag}_ed112_hd1024_rb2_q127"   "ega-${loss_tag}-ed112-hd1024-rb2-q127"   --override ega.encoded_dim=112   --override ega.hidden_dim=1024   --override ega.residual_blocks=2

run_case   "ega_${loss_tag}_ed096_hd1024_rb2_q127"   "ega-${loss_tag}-ed096-hd1024-rb2-q127"   --override ega.encoded_dim=96   --override ega.hidden_dim=1024   --override ega.residual_blocks=2

run_case   "ega_${loss_tag}_ed080_hd1024_rb2_q127"   "ega-${loss_tag}-ed080-hd1024-rb2-q127"   --override ega.encoded_dim=80   --override ega.hidden_dim=1024   --override ega.residual_blocks=2

run_case   "ega_${loss_tag}_ed112_hd1536_rb3_q127"   "ega-${loss_tag}-ed112-hd1536-rb3-q127"   --override ega.encoded_dim=112   --override ega.hidden_dim=1536   --override ega.residual_blocks=3

run_case   "ega_${loss_tag}_ed096_hd1536_rb3_q159"   "ega-${loss_tag}-ed096-hd1536-rb3-q159"   --override ega.encoded_dim=96   --override ega.hidden_dim=1536   --override ega.residual_blocks=3   --override ega.quantization_level=159
