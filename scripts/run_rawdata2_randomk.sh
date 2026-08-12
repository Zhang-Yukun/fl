#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=. python -m fedlab.entrypoints.train \
  --config configs/rawdata2_randomk.yaml \
  --mode federated \
  --override experiment.output_dir=outputs/rawdata2_randomk \
  "$@"
