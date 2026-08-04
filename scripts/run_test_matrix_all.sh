#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/test/logs

if tmux has-session -t test_gpu0 2>/dev/null; then
  tmux kill-session -t test_gpu0
fi
if tmux has-session -t test_gpu1 2>/dev/null; then
  tmux kill-session -t test_gpu1
fi

tmux new-session -d -s test_gpu0 "cd $(pwd) && bash scripts/run_test_matrix_gpu0.sh 2>&1 | tee outputs/test/logs/gpu0_master.log"
tmux new-session -d -s test_gpu1 "cd $(pwd) && bash scripts/run_test_matrix_gpu1.sh 2>&1 | tee outputs/test/logs/gpu1_master.log"

echo "Started tmux sessions: test_gpu0, test_gpu1"
echo "Queue logs: outputs/test/logs/gpu0_master.log and outputs/test/logs/gpu1_master.log"
echo "Attach with: tmux attach -t test_gpu0"
