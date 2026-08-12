#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=. python -m federated_ts.entrypoints.train \
  --config configs/rawdata2_ega.yaml \
  --mode federated \
  --override experiment.output_dir=outputs/rawdata2_ega \
  "$@"
