#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=. python -m fedlab.entrypoints.train \
  --config configs/qsgd.yaml \
  --mode federated \
  --override experiment.output_dir=outputs/qsgd \
  "$@"
