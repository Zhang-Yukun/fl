from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


CURRENT_SCRIPTS = (
    "run_analyze_experiment_suite_batch.sh",
    "run_controlled_suite.sh",
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
    "run_suite.sh",
)

REMOVED_SCRIPTS = (
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
        'FEDERATED_ALGORITHMS="${FEDERATED_ALGORITHMS:-fedavg,topk,ega}"',
        'TASK_SET="${TASK_SET:-rare}"',
        'TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS:-rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10}"',
        'TASK_CLIENT_IDS="${TASK_CLIENT_IDS:-rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3}"',
        'TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS:-rare}"',
        '--tasks task1,task2|all',
        'TASK_SET=rare,mnist,cifar10 bash SCRIPT --modes single_sync',
        'parse_named_map_keys() {',
        'lookup_named_map_value() {',
        'list_named_values() {',
        'selected_tasks() {',
        'task_config_path() {',
        'task_client_ids() {',
        'task_uses_loss_override() {',
        'mapfile -t TASK_LIST < <(selected_tasks "${TASK_SET}")',
        'for task in "${TASK_LIST[@]}"; do',
        'local outdir="${BASE_OUTPUT}/${task}/${run_name}"',
        'tracking_args "${task}-${tracking_name}"',
        'clients_raw="$(lookup_named_map_value "${TASK_CLIENT_IDS}" "${task}")"',
        'list_named_values "${clients_raw}"',
    ):
        assert marker in content


def test_run_suite_keeps_loss_override_only_for_rare_forecasting():
    content = _assert_executable("run_suite.sh")
    assert 'if task_uses_loss_override "${task}"; then' in content
    assert content.count('training.loss=%s') == 2


def test_controlled_suite_forwards_tasks_and_runs_single_base_suite():
    content = _assert_executable("run_controlled_suite.sh")
    for marker in (
        'TASK_SET="${TASK_SET:-rare}"',
        'TASK_SET=task1,task2|all',
        'TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS:-rare=configs/rare;mnist=configs/mnist;cifar10=configs/cifar10}"',
        'TASK_CLIENT_IDS="${TASK_CLIENT_IDS:-rare=Nd2O3,CeO2,La2O3;mnist=m1,m2,m3;cifar10=c1,c2,c3}"',
        'TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS:-rare}"',
        'PROFILE=noattack runs centralized + fedavg/topk/ega for the selected tasks.',
        'PROFILE=attack runs centralized + fedavg/topk/ega with attack enabled for the selected tasks.',
        'BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"',
        'TASK_SET="${TASK_SET}"',
        'TASK_CONFIG_DIRS="${TASK_CONFIG_DIRS}"',
        'TASK_CLIENT_IDS="${TASK_CLIENT_IDS}"',
        'TASK_LOSS_OVERRIDE_TASKS="${TASK_LOSS_OVERRIDE_TASKS}"',
        'FEDERATED_ALGORITHMS="${BASE_ALGOS}"',
        'bash scripts/run_suite.sh --modes "${SUITE_MODES}" --tasks "${TASK_SET}"',
    ):
        assert marker in content
    assert 'run_ega_matrix' not in content
    assert 'EVAL_MODE=' not in content
    assert content.count('bash scripts/run_suite.sh --modes') == 1


def test_seed_wrappers_delegate_to_controlled_suite():
    for path in sorted(SCRIPT_DIR.glob('run_exp_seed*.sh')):
        content = _assert_executable(path.name)
        assert 'bash scripts/run_controlled_suite.sh' in content
        assert 'BASE_ALGOS=fedavg,topk,ega' in content


def test_removed_scripts_are_absent():
    for name in REMOVED_SCRIPTS:
        assert not (SCRIPT_DIR / name).exists(), name


def test_batch_analysis_wrapper_supports_single_seed_and_multiseed_outputs():
    content = _assert_executable("run_analyze_experiment_suite_batch.sh")
    for marker in (
        'INPUT_ROOT="${INPUT_ROOT:-outputs/output/exp}"',
        'OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/exp}"',
        'MODE="${MODE:-single_sync}"',
        'PROFILE="${PROFILE:-noattack}"',
        'SEEDS_RAW="${SEEDS:-42 55 2026 8192}"',
        'LOSSES_RAW="${LOSSES:-mse mae}"',
        'ALGORITHMS_RAW="${ALGORITHMS:-centralized fedavg topk ega}"',
        'INCLUDE_OLD="${INCLUDE_OLD:-false}"',
        '--input-root PATH',
        '--output-root PATH',
        '--mode NAME',
        '--profile noattack|attack',
        '--seeds "42 55 2026 8192" | 42,55,2026,8192',
        '--losses "mse mae" | mse,mae',
        '--algorithms "centralized fedavg topk ega" | centralized,fedavg,topk,ega',
        'fedlab.tools.analyze_experiment_suite',
        'local input_dir="${INPUT_ROOT}/${MODE}/${seed}/${PROFILE}_${loss}"',
        'local output_dir="${OUTPUT_ROOT}/${MODE}/${seed}/${PROFILE}_${loss}"',
        'local output_dir="${OUTPUT_ROOT}/${MODE}/multiseed/${PROFILE}_${loss}"',
        'cmd+=("${INPUT_ROOT}/${MODE}/${seed}/${PROFILE}_${loss}")',
        'cmd+=("${ALGORITHM_LIST[@]}")',
        'PYTHONPATH=. "${cmd[@]}"',
    ):
        assert marker in content
