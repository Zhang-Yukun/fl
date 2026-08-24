#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run -n torch_env python -m fedlab.entrypoints.train \
  --config configs/secure_quantized_fedavg.yaml \
  --mode federated \
  "$@"
