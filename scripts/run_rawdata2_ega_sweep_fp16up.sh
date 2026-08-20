#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=. python -m fedlab.entrypoints.train \
  --config configs/rawdata2_ega.yaml \
  --mode federated \
  --override experiment.name=ega_sweep_fp16up \
  --override experiment.output_dir=outputs/ega_sweep_fp16up \
  --override ega.encoded_dtype=float16 \
  "$@"
