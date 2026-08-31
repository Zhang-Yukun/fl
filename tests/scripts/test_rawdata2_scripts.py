from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


CURRENT_SCRIPTS = (
    "run_analyze_experiment_suite_batch.sh",
    "run_replay_saved_update_batch.sh",
    "monitor_tcp_port_traffic.sh",
    "run_controlled_suite.sh",
    "run_exp_seed42.sh",
    "run_exp_seed42_part1.sh",
    "run_exp_seed42_part2.sh",
    "run_exp_seed2026.sh",
    "run_exp_seed2026_part1.sh",
    "run_exp_seed2026_part2.sh",
    "run_exp_seed4096.sh",
    "run_exp_seed4096_part1.sh",
    "run_exp_seed4096_part2.sh",
    "run_exp_seed8192.sh",
    "run_exp_seed8192_part1.sh",
    "run_exp_seed8192_part2.sh",
    "run_suite.sh",
)

REMOVED_SCRIPTS = (
    "run_exp_seed2026_mae.sh",
    "run_exp_seed2026_mae_part1.sh",
    "run_exp_seed2026_mae_part2.sh",
    "run_exp_seed2026_mse.sh",
    "run_exp_seed2026_mse_part1.sh",
    "run_exp_seed2026_mse_part2.sh",
    "run_exp_seed42_mae.sh",
    "run_exp_seed42_mae_part1.sh",
    "run_exp_seed42_mae_part2.sh",
    "run_exp_seed42_mse.sh",
    "run_exp_seed42_mse_part1.sh",
    "run_exp_seed42_mse_part2.sh",
    "run_exp_seed55_mae.sh",
    "run_exp_seed55_mae_part1.sh",
    "run_exp_seed55_mae_part2.sh",
    "run_exp_seed55_mse.sh",
    "run_exp_seed55_mse_part1.sh",
    "run_exp_seed55_mse_part2.sh",
    "run_exp_seed8192_mae.sh",
    "run_exp_seed8192_mae_part1.sh",
    "run_exp_seed8192_mae_part2.sh",
    "run_exp_seed8192_mse.sh",
    "run_exp_seed8192_mse_part1.sh",
    "run_exp_seed8192_mse_part2.sh",
    "run_tmux_three_task_suite.sh",
    "run_ega_noattack_adam_sweep_tmux.sh",
    "run_ega_noattack_adam_sweep_worker.sh",
    "run_ega_precision_adam_sweep_tmux.sh",
    "run_ega_precision_adam_sweep_worker.sh",
    "run_ega_targeted_noattack_alt_mae.sh",
    "run_ega_targeted_noattack_alt_mse.sh",
    "run_ega_targeted_noattack_tmux.sh",
    "run_ega_targeted_noattack_worker.sh",
    "run_oracle_gpu0_batch.sh",
    "run_oracle_gpu0_batch_adam.sh",
    "run_oracle_gpu0_ega_tune.sh",
    "run_oracle_gpu0_fast.sh",
    "run_oracle_gpu1_batch.sh",
    "run_oracle_gpu1_batch_adam.sh",
    "run_oracle_gpu1_ega_tune.sh",
    "run_oracle_gpu1_fast.sh",
    "run_oracle_gpu3_batch_adam.sh",
    "run_oracle_gpu3_ega_tune.sh",
    "run_oracle_gpu4_batch_adam.sh",
    "run_oracle_gpu4_ega_tune.sh",
    "run_oracle_noattack_mae_adam_single_sync.sh",
    "run_oracle_noattack_mae_adam_single_sync_gpu0.sh",
    "run_oracle_noattack_mae_adam_single_sync_gpu2.sh",
    "run_oracle_noattack_mse_adam_single_sync.sh",
    "run_oracle_noattack_mse_adam_single_sync_gpu1.sh",
    "run_oracle_noattack_mse_adam_single_sync_gpu3.sh",
    "run_oracle_suite.sh",
)


def _assert_executable(name: str) -> str:
    path = SCRIPT_DIR / name
    assert path.exists(), name
    assert path.stat().st_mode & 0o111, name
    content = path.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash")
    return content


def test_current_scripts_are_executable():
    for name in CURRENT_SCRIPTS:
        _assert_executable(name)


def test_run_suite_supports_task_matrix_and_default_four_algorithms():
    content = _assert_executable("run_suite.sh")
    for marker in (
        'RUN_TAG="${RUN_TAG:-suite}"',
        'TRACKING_TAG="${TRACKING_TAG:-suite}"',
        'RUNTIME_SEED="${RUNTIME_SEED:-${SUITE_SEED}}"',
        'QSGD_SEED="${QSGD_SEED:-${SUITE_SEED}}"',
        'RANDOMK_SEED="${RANDOMK_SEED:-${SUITE_SEED}}"',
        'ADAPTIVE_RDP_SEED="${ADAPTIVE_RDP_SEED:-${SUITE_SEED}}"',
        'QINT8_SEED="${QINT8_SEED:-${SUITE_SEED}}"',
        'EGA_QUANTIZATION_SEED="${EGA_QUANTIZATION_SEED:-${SUITE_SEED}}"',
        'EGA_PRETRAIN_SEED="${EGA_PRETRAIN_SEED:-${SUITE_SEED}}"',
        'FEDERATED_ALGORITHMS="${FEDERATED_ALGORITHMS:-fedavg,topk,ega}"',
        'TASK_SET="${TASK_SET:-rare}"',
        'TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS:-rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10}"',
        'TASK_CLIENT_IDS="${TASK_CLIENT_IDS:-rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3}"',
        'TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS:-rare}"',
        'TASK_IN_BASE_OUTPUT="${TASK_IN_BASE_OUTPUT:-false}"',
        '--tasks task1,task2|all',
        'TASK_SET=rare,mnist,cifar10 bash SCRIPT --modes single_sync',
        'parse_named_map_keys() {',
        'lookup_named_map_value() {',
        'list_named_values() {',
        'start_grpc_port_monitor() {',
        'stop_grpc_port_monitor() {',
        'bash scripts/monitor_tcp_port_traffic.sh',
        'grpc_port_traffic',
        'started grpc port monitor pid=${monitor_pid} address=${address} dir=${monitor_dir}" >&2',
        'selected_tasks() {',
        'task_config_path() {',
        'task_client_ids() {',
        'task_output_dir() {',
        'task_uses_loss_override() {',
        'algorithm_output_dir_name() {',
        'mapfile -t TASK_LIST < <(selected_tasks "${TASK_SET}")',
        'for task in "${TASK_LIST[@]}"; do',
        'output_name="$(algorithm_output_dir_name "${run_name}")"',
        'local outdir="$(task_output_dir "${task}")/${output_name}"',
        'tracking_args "${tracking_name}"',
        'clients_raw="$(lookup_named_map_value "${TASK_CLIENT_IDS}" "${task}")"',
        'list_named_values "${clients_raw}"',
    ):
        assert marker in content
    assert 'CAPTURE_FREQUENCY_ROUNDS="${CAPTURE_FREQUENCY_ROUNDS:-}"' in content
    assert "emit_optional_override 'replay_capture.frequency_rounds' \"${CAPTURE_FREQUENCY_ROUNDS}\"" in content


def test_run_suite_keeps_loss_override_only_for_rare_forecasting():
    content = _assert_executable("run_suite.sh")
    assert 'if task_uses_loss_override "${task}"; then' in content
    assert content.count('training.loss=%s') == 2


def test_controlled_suite_forwards_tasks_and_runs_single_base_suite():
    content = _assert_executable("run_controlled_suite.sh")
    for marker in (
        'CAPTURE_FREQUENCY_ROUNDS="${CAPTURE_FREQUENCY_ROUNDS:-}"',
        'TASK_SET="${TASK_SET:-rare}"',
        'TASK_SET=task1,task2|all',
        'TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS:-rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10}"',
        'TASK_CLIENT_IDS="${TASK_CLIENT_IDS:-rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3}"',
        'TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS:-rare}"',
        '训练脚本只负责 centralized + fedavg/topk/ega 训练与离线 replay 所需更新采集。',
        'TASK_IN_BASE_OUTPUT=true',
        'RUNTIME_SEED="${RUNTIME_SEED:-${SUITE_SEED}}"',
        'QSGD_SEED="${QSGD_SEED:-${SUITE_SEED}}"',
        'RANDOMK_SEED="${RANDOMK_SEED:-${SUITE_SEED}}"',
        'ADAPTIVE_RDP_SEED="${ADAPTIVE_RDP_SEED:-${SUITE_SEED}}"',
        'QINT8_SEED="${QINT8_SEED:-${SUITE_SEED}}"',
        'EGA_QUANTIZATION_SEED="${EGA_QUANTIZATION_SEED:-${SUITE_SEED}}"',
        'EGA_PRETRAIN_SEED="${EGA_PRETRAIN_SEED:-${SUITE_SEED}}"',
        '攻击执行由独立 replay 脚本完成，不再区分 noattack/attack profile。',
        'STARTUP_WAIT_SECONDS=60',
        'EGA_ARTIFACT_PATH=artifacts/ega/ega_h240_v1.pt',
        'EGA_PRETRAIN_DEVICE=same',
        'EGA_PRETRAIN_EPOCHS=220',
        'TASK_SET="${TASK_SET}"',
        'TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS}"',
        'TASK_CLIENT_IDS="${TASK_CLIENT_IDS}"',
        'TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS}"',
        'TASK_IN_BASE_OUTPUT="${TASK_IN_BASE_OUTPUT}"',
        'STARTUP_WAIT_SECONDS="${STARTUP_WAIT_SECONDS}"',
        'RUNTIME_SEED="${RUNTIME_SEED}"',
        'QSGD_SEED="${QSGD_SEED}"',
        'RANDOMK_SEED="${RANDOMK_SEED}"',
        'ADAPTIVE_RDP_SEED="${ADAPTIVE_RDP_SEED}"',
        'QINT8_SEED="${QINT8_SEED}"',
        'EGA_QUANTIZATION_SEED="${EGA_QUANTIZATION_SEED}"',
        'EGA_PRETRAIN_SEED="${EGA_PRETRAIN_SEED}"',
        'EGA_ARTIFACT_PATH="${EGA_ARTIFACT_PATH}"',
        'EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE}"',
        'EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS}"',
        'FEDERATED_ALGORITHMS="${BASE_ALGOS}"',
        'bash scripts/run_suite.sh --modes "${SUITE_MODES}" --tasks "${TASK_SET}" "${PASSTHROUGH_ARGS[@]}"',
    ):
        assert marker in content
    assert 'RUN_TAG="${RUN_TAG:-capturefreq${CAPTURE_FREQUENCY_ROUNDS}}"' in content
    assert 'RUN_TAG="${RUN_TAG:-suite}"' in content
    assert 'TRACKING_TAG="${TRACKING_TAG:-suite}"' in content
    assert 'run_ega_matrix' not in content
    assert 'EVAL_MODE=' not in content
    assert 'PASSTHROUGH_ARGS=("$@")' in content
    assert content.count('bash scripts/run_suite.sh --modes') == 1


def test_seed_wrappers_delegate_to_controlled_suite():
    for path in sorted(SCRIPT_DIR.glob('run_exp_seed*.sh')):
        content = _assert_executable(path.name)
        assert 'bash scripts/run_controlled_suite.sh' in content
        assert 'BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"' in content
        assert 'bash scripts/run_controlled_suite.sh "$@"' in content
        assert 'RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"' in content
        if '_part' in path.name:
            assert 'LOSSES=(mse)' in content
        if '_part1.sh' in path.name:
            assert 'MODE="single_sync"' in content
        if '_part2.sh' in path.name:
            assert 'MODE="multi_sync"' in content


def test_removed_scripts_are_absent():
    for name in REMOVED_SCRIPTS:
        assert not (SCRIPT_DIR / name).exists(), name


def test_port_traffic_monitor_script_captures_tcpdump_and_writes_summary():
    content = _assert_executable("monitor_tcp_port_traffic.sh")
    for marker in (
        'TCPDUMP_BIN="${TCPDUMP_BIN:-tcpdump}"',
        'INTERFACE="${INTERFACE:-any}"',
        '--port PORT',
        '--output-dir DIR',
        'RAW_LOG_NAME="${RAW_LOG_NAME:-grpc_port_traffic.tcpdump.log}"',
        'SUMMARY_NAME="${SUMMARY_NAME:-grpc_port_traffic.summary.json}"',
        'summarize_capture() {',
        'sent_payload_bytes',
        'received_payload_bytes',
        'tcpdump_stderr',
        '"${TCPDUMP_BIN}" -n -l -tt -i "${INTERFACE}" "tcp port ${PORT}"',
        'trap cleanup EXIT HUP INT TERM',
    ):
        assert marker in content


def test_batch_replay_wrapper_scans_saved_update_runs_and_splits_dlg_idlg_outputs():
    content = _assert_executable("run_replay_saved_update_batch.sh")
    for marker in (
        'INPUT_ROOT="${INPUT_ROOT:-outputs/exp}"',
        'ALGORITHMS_RAW="${ALGORITHMS:-fedavg topk ega}"',
        'TASKS_RAW="${TASKS:-}"',
        'MODES_RAW="${MODES:-}"',
        'SEEDS_RAW="${SEEDS:-}"',
        'LOSSES_RAW="${LOSSES:-}"',
        'RUN_DLG="${RUN_DLG:-true}"',
        'RUN_IDLG="${RUN_IDLG:-true}"',
        'SKIP_EXISTING="${SKIP_EXISTING:-true}"',
        '--dlg-only',
        '--idlg-only',
        '--force',
        '--override KEY=VALUE',
        'discover_run_dirs() {',
        "find \"${INPUT_ROOT}\" -type f -path '*/saved_updates/index.json' | sort",
        'run_method() {',
        'offline_attack_replay_${method}',
        'fedlab.tools.replay_saved_update_${method}',
        'if [[ "${SKIP_EXISTING}" == "true" && -f "${output_dir}/attack_summary.json" ]]; then',
        "IFS='/' read -r -a parts <<< \"${rel_path}\"",
        'if ! matches_filters "${task}" "${mode}" "${seed}" "${loss}" "${algorithm}"; then',
        'PYTHONPATH=. "${cmd[@]}"',
    ):
        assert marker in content


def test_batch_analysis_wrapper_supports_task_scoped_single_seed_and_multiseed_outputs():
    content = _assert_executable("run_analyze_experiment_suite_batch.sh")
    for marker in (
        'INPUT_ROOT="${INPUT_ROOT:-outputs/exp}"',
        'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/exp}"',
        'TASKS_RAW="${TASKS:-rare mnist cifar10}"',
        'TASK_LOSS_MAP="${TASK_LOSS_MAP:-rare=mse,mae;mnist=cross_entropy;cifar10=cross_entropy}"',
        'MODE="${MODE:-single_sync}"',
        'SEEDS_RAW="${SEEDS:-42 4096 2026 8192}"',
        'ALGORITHMS_RAW="${ALGORITHMS:-centralized fedavg topk ega}"',
        'INCLUDE_OLD="${INCLUDE_OLD:-false}"',
        '--input-root PATH',
        '--output-root PATH',
        '--tasks "rare mnist cifar10" | rare,mnist,cifar10',
        '--task-loss-map "rare=mse,mae;mnist=cross_entropy;cifar10=cross_entropy"',
        '--mode NAME',
        '--seeds "42 4096 2026 8192" | 42,4096,2026,8192',
        '--algorithms "centralized fedavg topk ega" | centralized,fedavg,topk,ega',
        'lookup_named_map_value() {',
        'task_losses() {',
        'read -r -a TASK_LIST <<< "$(parse_list "${TASKS_RAW}")"',
        'losses_raw="$(lookup_named_map_value "${TASK_LOSS_MAP}" "${task}")"',
        'local input_dir="${INPUT_ROOT}/${task}/${MODE}/${seed}/${loss}"',
        'local output_dir="${OUTPUT_ROOT}/${task}/${MODE}/${seed}/${loss}"',
        'local output_dir="${OUTPUT_ROOT}/${task}/${MODE}/multiseed/${loss}"',
        'cmd+=("${input_dir}")',
        'if [[ ! -d "${input_dir}" ]]; then',
        'if [[ ${found_inputs} -eq 0 ]]; then',
        'cmd+=("${ALGORITHM_LIST[@]}")',
        'PYTHONPATH=. "${cmd[@]}"',
    ):
        assert marker in content
