from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


RETAINED_SUITES = (
    "run_oracle_suite.sh",
    "run_suite.sh",
)

RETAINED_WRAPPERS = (
    "run_oracle_gpu0_batch.sh",
    "run_oracle_gpu1_batch.sh",
    "run_oracle_gpu0_fast.sh",
    "run_oracle_gpu1_fast.sh",
    "run_oracle_gpu0_batch_adam.sh",
    "run_oracle_gpu1_batch_adam.sh",
    "run_oracle_noattack_mse_adam_single_sync.sh",
    "run_oracle_noattack_mse_adam_single_sync_gpu1.sh",
    "run_oracle_noattack_mse_adam_single_sync_gpu3.sh",
    "run_oracle_noattack_mae_adam_single_sync.sh",
    "run_oracle_noattack_mae_adam_single_sync_gpu0.sh",
    "run_oracle_noattack_mae_adam_single_sync_gpu2.sh",
    "run_controlled_suite.sh",
    "run_exp_seed42_mse.sh",
    "run_exp_seed42_mae.sh",
    "run_exp_seed2026_mse.sh",
    "run_exp_seed2026_mae.sh",
    "run_exp_seed55_mse.sh",
    "run_exp_seed55_mae.sh",
    "run_exp_seed8192_mse.sh",
    "run_exp_seed8192_mae.sh",
    "run_exp_seed42_mse_part1.sh",
    "run_exp_seed42_mse_part2.sh",
    "run_exp_seed42_mae_part1.sh",
    "run_exp_seed42_mae_part2.sh",
    "run_exp_seed2026_mse_part1.sh",
    "run_exp_seed2026_mse_part2.sh",
    "run_exp_seed2026_mae_part1.sh",
    "run_exp_seed2026_mae_part2.sh",
    "run_exp_seed55_mse_part1.sh",
    "run_exp_seed55_mse_part2.sh",
    "run_exp_seed55_mae_part1.sh",
    "run_exp_seed55_mae_part2.sh",
    "run_exp_seed8192_mse_part1.sh",
    "run_exp_seed8192_mse_part2.sh",
    "run_exp_seed8192_mae_part1.sh",
    "run_exp_seed8192_mae_part2.sh",
    "run_analyze_experiment_suite_batch.sh",
)

RETAINED_EGA_SWEEPS = (
    "run_ega_noattack_adam_sweep_tmux.sh",
    "run_ega_noattack_adam_sweep_worker.sh",
    "run_ega_precision_adam_sweep_tmux.sh",
    "run_ega_precision_adam_sweep_worker.sh",
)

REMOVED_LEGACY = (
    "run_fedavg_consistency_check.sh",
    "run_rawdata2_adaptive_clipped_rdp_fedavg_deterministic.sh",
    "run_rawdata2_all.sh",
    "run_rawdata2_centralized.sh",
    "run_rawdata2_ega.sh",
    "run_rawdata2_ega_formal.sh",
    "run_rawdata2_fedavg.sh",
    "run_rawdata2_fedlab_topk.sh",
    "run_rawdata2_qsgd.sh",
    "run_rawdata2_randomk.sh",
    "run_rawdata2_secure_quantized_fedavg.sh",
    "run_rawdata2_sign.sh",
    "run_repro_pat50_centralized.sh",
    "run_repro_pat50_fedavg.sh",
    "run_repro_pat50_qfloat16_bidir.sh",
    "run_repro_pat50_qint8_bidir.sh",
    "run_repro_pat50_topk_fedavg.sh",
    "run_test_matrix_all.sh",
    "run_test_matrix_common.sh",
    "run_test_matrix_gpu0.sh",
    "run_test_matrix_gpu1.sh",
    "run_ega_target_sweep_tmux.sh",
    "run_ega_target_sweep_worker.sh",
    "run_formal_adaptive_clipped_rdp_suite.sh",
    "run_formal_centralized_fedavg_qint8_suite.sh",
    "run_formal_multi_algo_suite_tmux.sh",
    "run_formal_multi_algo_worker.sh",
    "run_rawdata2_ega_sweep_base.sh",
    "run_rawdata2_ega_sweep_ed160.sh",
    "run_rawdata2_ega_sweep_ed160_q255.sh",
    "run_rawdata2_ega_sweep_ed192.sh",
    "run_rawdata2_ega_sweep_ed288.sh",
    "run_rawdata2_ega_sweep_fp16up.sh",
    "run_rawdata2_ega_sweep_q255.sh",
    "run_suite_500r_tmux.sh",
    "run_suite_500r_worker.sh",
)


def _assert_executable(name: str) -> str:
    path = SCRIPT_DIR / name
    assert path.exists(), name
    assert path.stat().st_mode & 0o111, name
    content = path.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env bash")
    return content


def test_retained_suite_and_wrapper_scripts_are_executable():
    for name in (*RETAINED_SUITES, *RETAINED_WRAPPERS):
        _assert_executable(name)


def test_retained_ega_sweep_scripts_are_executable():
    for name in RETAINED_EGA_SWEEPS:
        _assert_executable(name)


def test_oracle_suite_runs_ega_after_fedavg_and_uses_env_parameters():
    content = _assert_executable("run_oracle_suite.sh")
    fedavg_pos = content.index('"fedavg_${mode}_uupdate_dmodel_${RUN_TAG}"')
    ega_pos = content.index('local ega_run_name="${ega_label}_${mode}_uupdate_dmodel_${RUN_TAG}"')
    topk_pos = content.index('"topk_${mode}_uupdate_dmodel_${RUN_TAG}"')
    assert fedavg_pos < ega_pos < topk_pos
    assert '--algorithms fedavg,topk,ega' in content
    for marker in (
        'RUN_TAG="${RUN_TAG:-oracle_attackfreq5}"',
        'TRACKING_TAG="${TRACKING_TAG:-oracle-attackfreq5}"',
        'SUITE_SEED="${SUITE_SEED:-2026}"',
        'RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"',
        'RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"',
        'ATTACK_ENABLED="${ATTACK_ENABLED:-true}"',
        'ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS:-5}"',
        'ATTACK_MAX_SAMPLES="${ATTACK_MAX_SAMPLES:-}"',
        'ATTACK_SEED="${ATTACK_SEED:-${SUITE_SEED}}"',
        'LOSS_NAME="${LOSS_NAME:-mse}"',
        'LOSS_TAG="${LOSS_TAG:-${LOSS_NAME}}"',
        'TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-}"',
        'TRAIN_MOMENTUM="${TRAIN_MOMENTUM:-}"',
        'TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-}"',
        'TRAIN_OPTIMIZER_EPS="${TRAIN_OPTIMIZER_EPS:-}"',
        'EVAL_MODE="${EVAL_MODE:-protocol}"',
        'SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-true}"',
        'MODEL_DROPOUT="${MODEL_DROPOUT:-0.1}"',
        'runtime.seed=%s',
        'runtime.num_threads=1',
        'runtime.num_interop_threads=1',
        'FEDERATED_ALGORITHMS="${FEDERATED_ALGORITHMS:-fedavg,topk,ega}"',
        'TOPK_FRACTION="${TOPK_FRACTION:-}"',
        'QSGD_LEVELS="${QSGD_LEVELS:-}"',
        'QSGD_SEED="${QSGD_SEED:-${SUITE_SEED}}"',
        'RANDOMK_FRACTION="${RANDOMK_FRACTION:-}"',
        'QINT8_DTYPE="${QINT8_DTYPE:-}"',
        'EGA_DOWNLOAD_METHOD="${EGA_DOWNLOAD_METHOD:-}"',
        'EGA_QUANTIZATION_SEED="${EGA_QUANTIZATION_SEED:-${SUITE_SEED}}"',
        'EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE:-}"',
        'EGA_ENCODED_DIM="${EGA_ENCODED_DIM:-240}"',
        'EGA_HIDDEN_DIM="${EGA_HIDDEN_DIM:-1024}"',
        'EGA_RESIDUAL_BLOCKS="${EGA_RESIDUAL_BLOCKS:-2}"',
        'EGA_QUANTIZATION_LEVEL="${EGA_QUANTIZATION_LEVEL:-127}"',
        'EGA_NORMALIZATION_EMA="${EGA_NORMALIZATION_EMA:-0.9}"',
        'EGA_PRETRAIN_EPOCHS="${EGA_PRETRAIN_EPOCHS:-100}"',
        'emit_optional_override() {',
        'attack.enabled=%s',
        'attack.max_samples=%s',
        'attack.seed=%s',
        'training.loss=%s',
        'training.optimizer=%s',
        'training.momentum=%s',
        'training.weight_decay=%s',
        'training.optimizer_eps=%s',
        'evaluation.mode=%s',
        'data.shuffle_train=%s',
        'model.dropout=%s',
        'effective_run_name() {',
        'effective_tracking_name() {',
        "printf '%s%s_seed%s%s\\n'",
        "printf '%s-seed%s\\n'",
        'ega_name_signature() {',
        'effective_ega_label() {',
        'ed%s_hd%s_rb%s_q%s_ema%s_pt%s',
        'ega_%s',
        'federated.topk_fraction',
        'federated.qsgd_levels',
        'federated.quantization_seed',
        'federated.quantization_dtype',
        'ega.download_method',
        'TRAIN_OPTIMIZER=adam TRAIN_LR=0.001 bash SCRIPT --modes centralized,single_sync',
        'EVAL_MODE=protocol SHUFFLE_TRAIN=true MODEL_DROPOUT=0.1 bash SCRIPT --modes single_sync',
        'FEDERATED_ALGORITHMS=fedavg,ega bash SCRIPT --modes single_sync',
        'RUNTIME_DEVICE=cuda:1 bash SCRIPT --modes single_sync',
        'SUITE_SEED=42 bash SCRIPT --modes single_sync',
        'RUN_NAME_PREFIX=debug_ RUN_NAME_SUFFIX=_trial1 bash SCRIPT --modes single_sync',
        'EGA_DOWNLOAD_METHOD=dense EGA_PRETRAIN_DEVICE=cuda:1 bash SCRIPT --modes single_sync',
    ):
        assert marker in content
    assert 'ega.download_method=ega' not in content
    assert 'topk10-' not in content
    assert 'qsgd63-' not in content
    assert 'ega.artifact_path' in content
    assert 'rm -f "${ega_artifact}"' not in content


def test_controlled_suite_wrapper_exposes_requested_matrix_controls():
    content = _assert_executable("run_controlled_suite.sh")
    for marker in (
        'PROFILE="${PROFILE:-noattack}"',
        'LOSS_NAME="${LOSS_NAME:-mse}"',
        'MODE_SET="${MODE_SET:-all}"',
        'SUITE_SEED="${SUITE_SEED:-2026}"',
        'RUNTIME_DEVICE="${RUNTIME_DEVICE:-cuda:0}"',
        'RUN_CENTRALIZED="${RUN_CENTRALIZED:-true}"',
        'BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-outputs/suite_${PROFILE}_${LOSS_NAME}_seed${SUITE_SEED}_$(date +%Y%m%d_%H%M%S)}"',
        'PROJECT_NAME="${PROJECT_NAME:-rare-earth-fl-suite-${PROFILE}-${LOSS_NAME}}"',
        'BASE_ALGOS="${BASE_ALGOS:-all}"',
        'BASE_ALGOS="${BASE_ALGOS:-fedavg,topk,ega}"',
        'MAPPED_MODES="$(map_modes "${MODE_SET}")"',
        'SUITE_MODES="$(suite_modes "${MAPPED_MODES}")"',
        "printf 'centralized,%s\\n' \"${federated_modes}\"",
        'bash scripts/run_suite.sh --modes "${SUITE_MODES}"',
        'bash scripts/run_suite.sh --modes "${MAPPED_MODES}"',
        'FEDERATED_ALGORITHMS=ega',
        'EGA_PRETRAIN_DEVICE="${RUNTIME_DEVICE}"',
    ):
        assert marker in content

    assert 'PROFILE=attack runs centralized + fedavg/topk/ega in the base suite' in content
    assert 'PROFILE=noattack runs centralized + all federated algorithms in the base suite' in content


def test_controlled_suite_wrapper_contains_shortlisted_ega_configs_for_mse_and_mae():
    content = _assert_executable("run_controlled_suite.sh")
    for marker in (
        'EGA_ENCODED_DIM=160',
        'EGA_HIDDEN_DIM=1024',
        'EGA_RESIDUAL_BLOCKS=2',
        'EGA_QUANTIZATION_LEVEL=159',
        'EGA_NORMALIZATION_EMA=0.95',
        'EGA_PRETRAIN_EPOCHS=150',
        'EGA_PRETRAIN_PATIENCE=30',
        'EGA_PRETRAIN_LR=0.0003',
        'EGA_ENCODED_DIM=168',
        'EGA_HIDDEN_DIM=2048',
        'EGA_RESIDUAL_BLOCKS=4',
        'EGA_NORMALIZATION_EMA=0.98',
        'EGA_PRETRAIN_EPOCHS=220',
        'EGA_PRETRAIN_PATIENCE=44',
        'EGA_PRETRAIN_TRAIN_GROUPS=50000',
        'EGA_PRETRAIN_VAL_GROUPS=25000',
        'EGA_HIDDEN_DIM=1536',
        'EGA_RESIDUAL_BLOCKS=3',
        'EGA_QUANTIZATION_LEVEL=127',
        'EGA_PRETRAIN_EPOCHS=200',
        'EGA_PRETRAIN_PATIENCE=40',
        'EGA_PRETRAIN_EPOCHS=180',
        'EGA_PRETRAIN_PATIENCE=36',
        'EGA_PRETRAIN_LR=0.00025',
        'EGA_PRETRAIN_TRAIN_GROUPS=40000',
        'EGA_PRETRAIN_VAL_GROUPS=20000',
    ):
        assert marker in content


def test_oracle_gpu_wrappers_call_generic_suite_without_duplicate_attackfreq5_runs():
    for name in RETAINED_WRAPPERS:
        content = _assert_executable(name)
        if name == "run_controlled_suite.sh":
            assert "run_suite.sh" in content
            continue
        if name == "run_analyze_experiment_suite_batch.sh":
            assert "fedlab.tools.analyze_experiment_suite" in content
            continue
        if name.startswith("run_exp_seed"):
            assert "run_controlled_suite.sh" in content
            assert 'PROJECT_NAME="re_fl_noattack_${mode}_adam"' in content
            assert 'BASE_ALGOS=fedavg,topk,ega' in content
            assert 'PROJECT_NAME="re_fl_attack_${mode}_adam"' in content
            if name.endswith("_part2.sh"):
                assert 'BASE_PORT="${BASE_PORT:-' in content
                assert 'BASE_PORT="${BASE_PORT}"' in content
            continue
        assert "run_oracle_suite.sh" in content
        assert "run_oracle_suite_param.sh" not in content
        assert "_mae.sh" not in content
        assert "LOSS_NAME=mse" in content or "LOSS_NAME=mae" in content
        assert "LOSS_TAG=mse" in content or "LOSS_TAG=mae" in content

    gpu0_batch = (SCRIPT_DIR / "run_oracle_gpu0_batch.sh").read_text(encoding="utf-8")
    gpu1_batch = (SCRIPT_DIR / "run_oracle_gpu1_batch.sh").read_text(encoding="utf-8")
    gpu0_fast = (SCRIPT_DIR / "run_oracle_gpu0_fast.sh").read_text(encoding="utf-8")
    gpu1_fast = (SCRIPT_DIR / "run_oracle_gpu1_fast.sh").read_text(encoding="utf-8")

    assert "RUN_TAG=oracle_attackfreq5 " not in gpu0_batch
    assert "RUN_TAG=oracle_attackfreq5 " not in gpu1_batch
    assert "RUN_TAG=oracle_attackfreq5 TRACKING_TAG=oracle-attackfreq5" in gpu0_fast
    assert "RUN_TAG=oracle_attackfreq5 TRACKING_TAG=oracle-attackfreq5" in gpu1_fast
    assert "GPU_ID=" not in gpu0_batch
    assert "GPU_ID=" not in gpu1_batch
    assert "GPU_ID=" not in gpu0_fast
    assert "GPU_ID=" not in gpu1_fast
    assert "CUDA_VISIBLE_DEVICES" not in (SCRIPT_DIR / "run_oracle_suite.sh").read_text(encoding="utf-8")
    assert 'RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"' in (SCRIPT_DIR / "run_oracle_suite.sh").read_text(encoding="utf-8")
    assert 'RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"' in (SCRIPT_DIR / "run_oracle_suite.sh").read_text(encoding="utf-8")


def test_removed_legacy_scripts_are_absent():
    for name in REMOVED_LEGACY:
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
