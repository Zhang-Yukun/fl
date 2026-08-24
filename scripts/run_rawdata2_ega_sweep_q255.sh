#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=. python -m fedlab.entrypoints.train \
  --config configs/ega.yaml \
  --mode federated \
  --override experiment.name=ega_sweep_q255 \
  --override experiment.output_dir=outputs/ega_sweep_q255 \
  --override ega.quantization_level=255 \
  "$@"
