#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-conda run -n torch_env python}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
RUNTIME_DEVICE="${RUNTIME_DEVICE:?RUNTIME_DEVICE is required}"
BASE_OUTPUT="${BASE_OUTPUT:?BASE_OUTPUT is required}"
WORKER_KIND="${WORKER_KIND:?WORKER_KIND is required}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-outputs/suite500_20260820_161030/fedavg_single_sync_500r_pat50/summary.json}"

run_case() {
  local run_name="$1"
  shift
  local outdir="${BASE_OUTPUT}/${run_name}"
  echo "[$(date '+%F %T')] start ${run_name} on gpu ${GPU_ID}"
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
    --override "tracking.enabled=false" \
    --override "attack.enabled=false" \
    --override "evaluation.mode=protocol" \
    --override "federated.rounds=500" \
    --override "training.patience=50" \
    --override "training.min_delta=0.0" \
    "$@"
  echo "[$(date '+%F %T')] finish ${run_name}"
  echo "[$(date '+%F %T')] compare ${run_name}"
  BASELINE_SUMMARY="${BASELINE_SUMMARY}" CANDIDATE_SUMMARY="${outdir}/summary.json" python3 - <<'PY'
import json, os
base = json.load(open(os.environ['BASELINE_SUMMARY'], 'r', encoding='utf-8'))
cand = json.load(open(os.environ['CANDIDATE_SUMMARY'], 'r', encoding='utf-8'))
base_mse = float(base['test']['mse'])
base_comm = float(base['total_parameter_bytes'])
cand_test = float(cand['test']['mse'])
cand_protocol = float(cand.get('protocol_test', {}).get('mse', cand_test))
cand_comm = float(cand['total_parameter_bytes'])
print({
    'candidate': os.path.basename(os.path.dirname(os.environ['CANDIDATE_SUMMARY'])),
    'rounds': cand['rounds'],
    'comm_ratio_vs_fedavg': cand_comm / base_comm,
    'comm_target_max': 1/8,
    'test_delta_pct': (cand_test - base_mse) / base_mse * 100.0,
    'protocol_delta_pct': (cand_protocol - base_mse) / base_mse * 100.0,
})
PY
}

common_overrides=(
  --override transport.upload_mode=update
  --override transport.download_mode=model
  --override ega.download_method=ega
  --override ega.download_dtype=float32
  --override ega.download_encoded_dtype=int8
  --override ega.download_encoded_stochastic_rounding=false
  --override ega.download_trainable_only=true
  --override ega.download_quantization_level=127
  --override ega.download_min_normalization=1e-6
  --override ega.download_predictive_coding=true
  --override ega.normalization_strategy=ema_reported_client_max_abs
  --override ega.normalization_ema=0.9
  --override ega.quantization_level=127
  --override ega.encoded_dtype=int8
  --override ega.encoded_stochastic_rounding=false
  --override ega.error_feedback=true
)

if [[ "${WORKER_KIND}" == "gpu0" ]]; then
  run_case ega_ed160_dm_ega_pc_q127 \
    "${common_overrides[@]}" \
    --override ega.artifact_path=artifacts/ega/ega_ed160_dm_ega_pc_q127.pt \
    --override ega.encoded_dim=160 \
    --override ega.hidden_dim=1024 \
    --override ega.residual_blocks=2 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case ega_ed128_dm_ega_pc_q127 \
    "${common_overrides[@]}" \
    --override ega.artifact_path=artifacts/ega/ega_ed128_dm_ega_pc_q127.pt \
    --override ega.encoded_dim=128 \
    --override ega.hidden_dim=1024 \
    --override ega.residual_blocks=2 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case ega_ed112_dm_ega_pc_q127 \
    "${common_overrides[@]}" \
    --override ega.artifact_path=artifacts/ega/ega_ed112_dm_ega_pc_q127.pt \
    --override ega.encoded_dim=112 \
    --override ega.hidden_dim=1024 \
    --override ega.residual_blocks=2 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}
else
  run_case ega_ed144_dm_ega_pc_q127 \
    "${common_overrides[@]}" \
    --override ega.artifact_path=artifacts/ega/ega_ed144_dm_ega_pc_q127.pt \
    --override ega.encoded_dim=144 \
    --override ega.hidden_dim=1024 \
    --override ega.residual_blocks=2 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case ega_ed096_dm_ega_pc_q127 \
    "${common_overrides[@]}" \
    --override ega.artifact_path=artifacts/ega/ega_ed096_dm_ega_pc_q127.pt \
    --override ega.encoded_dim=96 \
    --override ega.hidden_dim=1024 \
    --override ega.residual_blocks=2 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}

  run_case ega_ed128_dm_ega_pc_q095 \
    "${common_overrides[@]}" \
    --override ega.artifact_path=artifacts/ega/ega_ed128_dm_ega_pc_q095.pt \
    --override ega.encoded_dim=128 \
    --override ega.hidden_dim=1024 \
    --override ega.residual_blocks=2 \
    --override ega.quantization_level=95 \
    --override ega.download_quantization_level=95 \
    --override ega.pretrain.device=${RUNTIME_DEVICE}
fi
