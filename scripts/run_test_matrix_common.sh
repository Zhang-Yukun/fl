#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="$1"
BATCH_NAME="$2"
shift 2

BASE_OUTPUT="outputs/test"
LOG_DIR="${BASE_OUTPUT}/logs"
mkdir -p "$LOG_DIR"

run_train() {
  local config="$1"
  local mode="$2"
  local outdir="$3"
  shift 3

  echo "[$(date '+%F %T')] START ${outdir}" | tee -a "${LOG_DIR}/${BATCH_NAME}.log"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n torch_env python -m scripts.train     --config "${config}"     --mode "${mode}"     --override "experiment.output_dir=${outdir}"     --override "runtime.device=cuda:0"     --override "runtime.seed=2026"     --override "runtime.deterministic=true"     --override "training.patience=20"     --override "attack.seed=2026"     "$@" 2>&1 | tee "${LOG_DIR}/$(basename "${outdir}").log"
  echo "[$(date '+%F %T')] END ${outdir}" | tee -a "${LOG_DIR}/${BATCH_NAME}.log"
}
