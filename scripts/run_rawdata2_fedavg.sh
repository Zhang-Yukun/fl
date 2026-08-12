#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run -n torch_env python -m fedlab.entrypoints.train \
  --config configs/rawdata2_patchtst.yaml \
  --mode federated \
  --override experiment.mode=federated \
  --override experiment.output_dir=outputs/rawdata2_patchtst_fedavg \
  --override federated.algorithm=fedavg \
  "$@"
