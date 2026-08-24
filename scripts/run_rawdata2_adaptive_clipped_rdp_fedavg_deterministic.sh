#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/adaptive_clipped_rdp_fedavg_deterministic}"
SUITE_ROUNDS="${SUITE_ROUNDS:-}"
ATTACK_FREQUENCY="${ATTACK_FREQUENCY:-}"

EXTRA_OVERRIDES=()
if [[ -n "${SUITE_ROUNDS}" ]]; then
  EXTRA_OVERRIDES+=(--override "federated.rounds=${SUITE_ROUNDS}")
fi
if [[ -n "${ATTACK_FREQUENCY}" ]]; then
  EXTRA_OVERRIDES+=(--override "attack.frequency_rounds=${ATTACK_FREQUENCY}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=. "${PYTHON_BIN}" -m fedlab.entrypoints.train \
  --config configs/adaptive_clipped_rdp_fedavg_deterministic.yaml \
  --mode federated \
  --override "experiment.output_dir=${OUTPUT_DIR}" \
  --override "runtime.device=${RUNTIME_DEVICE}" \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"
