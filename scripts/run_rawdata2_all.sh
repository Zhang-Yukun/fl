#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

runs=(
  "scripts/run_rawdata2_centralized.sh"
  "scripts/run_rawdata2_fedavg.sh"
  "scripts/run_rawdata2_fedaware.sh"
)

for run_script in "${runs[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting ${run_script}"
  bash "${run_script}" "$@"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished ${run_script}"
  echo
done
