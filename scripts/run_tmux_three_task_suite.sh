#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TMUX_SESSION_PREFIX="${TMUX_SESSION_PREFIX:-ydxt_suite_2026}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tmux_three_task_suite_seed2026_r10}"
TASK_SET="${TASK_SET:-rare,mnist,cifar10}"
LOSS_NAME="${LOSS_NAME:-mse}"
SUITE_SEED="${SUITE_SEED:-2026}"
ROUNDS="${ROUNDS:-10}"
PATIENCE="${PATIENCE:-500}"
ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS:-10}"
TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-adam}"
GPU_SINGLE_NOATTACK="${GPU_SINGLE_NOATTACK:-0}"
GPU_SINGLE_ATTACK="${GPU_SINGLE_ATTACK:-1}"
GPU_MULTI_NOATTACK="${GPU_MULTI_NOATTACK:-0}"
GPU_MULTI_ATTACK="${GPU_MULTI_ATTACK:-1}"
BASE_PORT_MULTI_NOATTACK="${BASE_PORT_MULTI_NOATTACK:-58000}"
BASE_PORT_MULTI_ATTACK="${BASE_PORT_MULTI_ATTACK:-58100}"
STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS:-60}"
EGA_ARTIFACT_PATH="${EGA_ARTIFACT_PATH:-}"
EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE:-same}"
EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS:-100}"
TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS:-rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10}"
TASK_CLIENT_IDS="${TASK_CLIENT_IDS:-rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3}"
TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS:-rare}"

usage() {
  cat <<'USAGE'
Usage:
  TMUX_SESSION_PREFIX=ydxt_suite_2026   OUTPUT_ROOT=outputs/tmux_three_task_suite_seed2026_r10   TASK_SET=rare,mnist,cifar10   LOSS_NAME=mse   SUITE_SEED=2026   ROUNDS=10   PATIENCE=500   ATTACK_FREQUENCY_ROUNDS=10   TRAIN_OPTIMIZER=adam   GPU_SINGLE_NOATTACK=0   GPU_SINGLE_ATTACK=1   GPU_MULTI_NOATTACK=0   GPU_MULTI_ATTACK=1   BASE_PORT_MULTI_NOATTACK=58000   BASE_PORT_MULTI_ATTACK=58100   STARTUP_WAIT_SECONDS=60   EGA_ARTIFACT_PATH=artifacts/ega/ega_h240_v1.pt   EGA_PRETRAIN_DEVICE=same   EGA_PRETRAIN_EPOCHS=100   bash scripts/run_tmux_three_task_suite.sh
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"

launch_session() {
  local session_name="$1"
  local profile="$2"
  local mode_set="$3"
  local gpu_id="$4"
  local base_port="$5"
  local run_root="$6"

  local command
  printf -v command 'cd %q && env PROFILE=%q LOSS_NAME=%q TASK_SET=%q TASK_CONFIG_DIRS=%q TASK_CLIENT_IDS=%q TASK_LOSS_OVERRIDE_TASKS=%q MODE_SET=%q SUITE_SEED=%q RUNTIME_DEVICE=%q ROUNDS=%q PATIENCE=%q ATTACK_FREQUENCY_ROUNDS=%q TRAIN_OPTIMIZER=%q BASE_PORT=%q STARTUP_WAIT_SECONDS=%q EGA_ARTIFACT_PATH=%q EGA_PRETRAIN_DEVICE=%q EGA_PRETRAIN_EPOCHS=%q BASE_OUTPUT_ROOT=%q PROJECT_NAME=%q bash scripts/run_controlled_suite.sh' \
    "$(pwd)" \
    "${profile}" \
    "${LOSS_NAME}" \
    "${TASK_SET}" \
    "${TASK_CONFIG_DIRS}" \
    "${TASK_CLIENT_IDS}" \
    "${TASK_LOSS_OVERRIDE_TASKS}" \
    "${mode_set}" \
    "${SUITE_SEED}" \
    "cuda:${gpu_id}" \
    "${ROUNDS}" \
    "${PATIENCE}" \
    "${ATTACK_FREQUENCY_ROUNDS}" \
    "${TRAIN_OPTIMIZER}" \
    "${base_port}" \
    "${STARTUP_WAIT_SECONDS}" \
    "${EGA_ARTIFACT_PATH}" \
    "${EGA_PRETRAIN_DEVICE}" \
    "${EGA_PRETRAIN_EPOCHS}" \
    "${run_root}" \
    "${TMUX_SESSION_PREFIX}_${profile}_${mode_set}"

  tmux new-session -d -s "${session_name}" "conda run -n torch_env bash -lc $(printf %q "${command}")"
  echo "started ${session_name} -> ${run_root}"
}

launch_session "${TMUX_SESSION_PREFIX}_single_noattack" noattack single_sync "${GPU_SINGLE_NOATTACK}" 0 "${OUTPUT_ROOT}/single/noattack"
launch_session "${TMUX_SESSION_PREFIX}_single_attack" attack single_sync "${GPU_SINGLE_ATTACK}" 0 "${OUTPUT_ROOT}/single/attack"
launch_session "${TMUX_SESSION_PREFIX}_multi_noattack" noattack multi_sync "${GPU_MULTI_NOATTACK}" "${BASE_PORT_MULTI_NOATTACK}" "${OUTPUT_ROOT}/multi/noattack"
launch_session "${TMUX_SESSION_PREFIX}_multi_attack" attack multi_sync "${GPU_MULTI_ATTACK}" "${BASE_PORT_MULTI_ATTACK}" "${OUTPUT_ROOT}/multi/attack"

tmux list-sessions
