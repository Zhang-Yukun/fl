#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run -n torch_env python -m fedlab.entrypoints.train \
  --config configs/rawdata2_centralized.yaml \
  --mode centralized \
  --override experiment.mode=centralized \
  --override experiment.output_dir=outputs/rawdata2_centralized \
  "$@"
