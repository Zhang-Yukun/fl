#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

conda run -n torch_env python -m federated_ts.entrypoints.train \
  --config configs/rawdata2_secure_quantized_fedavg.yaml \
  --mode federated \
  "$@"
