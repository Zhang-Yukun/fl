#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=. python -m fedlab.entrypoints.train \
  --config configs/ega.yaml \
  --mode federated \
  --override experiment.name=ega_sweep_ed160 \
  --override experiment.output_dir=outputs/ega_sweep_ed160 \
  --override ega.encoded_dim=160 \
  "$@"
