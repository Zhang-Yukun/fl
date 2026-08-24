#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run -n torch_env python -m fedlab.entrypoints.train \
  --config configs/fedavg.yaml \
  --mode federated \
  --override experiment.mode=federated \
  --override experiment.output_dir=outputs/fedavg \
  --override federated.algorithm=fedavg \
  "$@"
