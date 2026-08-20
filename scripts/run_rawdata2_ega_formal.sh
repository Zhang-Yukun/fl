#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=. python -m fedlab.entrypoints.train \
  --config configs/rawdata2_ega.yaml \
  --mode federated \
  --override experiment.name=ega_formal \
  --override experiment.output_dir=outputs/ega_formal \
  "$@"
