#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:?RUNTIME_DEVICE is required}"
BASE_OUTPUT="${BASE_OUTPUT:?BASE_OUTPUT is required}"
LOSS_NAME="${LOSS_NAME:?LOSS_NAME is required}"
PROJECT_NAME="${PROJECT_NAME:-rare-earth-ega-adam-precision-explore-v1}"
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

run_case   "ega_${loss_tag}_lr5e4_r700_ed128_q127"   "ega-${loss_tag}-lr5e4-r700-ed128-q127"   --override training.lr=0.0005   --override federated.rounds=700   --override training.patience=70   --override ega.encoded_dim=128   --override ega.hidden_dim=1024   --override ega.residual_blocks=2   --override ega.quantization_level=127

run_case   "ega_${loss_tag}_lr5e4_r1000_ed128_q127"   "ega-${loss_tag}-lr5e4-r1000-ed128-q127"   --override training.lr=0.0005   --override federated.rounds=1000   --override training.patience=100   --override ega.encoded_dim=128   --override ega.hidden_dim=1024   --override ega.residual_blocks=2   --override ega.quantization_level=127

run_case   "ega_${loss_tag}_lr1e3_r500_ed160_q159"   "ega-${loss_tag}-lr1e3-r500-ed160-q159"   --override training.lr=0.001   --override federated.rounds=500   --override training.patience=50   --override ega.encoded_dim=160   --override ega.hidden_dim=1024   --override ega.residual_blocks=2   --override ega.quantization_level=159

run_case   "ega_${loss_tag}_lr5e4_r700_ed192_q191"   "ega-${loss_tag}-lr5e4-r700-ed192-q191"   --override training.lr=0.0005   --override federated.rounds=700   --override training.patience=70   --override ega.encoded_dim=192   --override ega.hidden_dim=1536   --override ega.residual_blocks=3   --override ega.quantization_level=191
