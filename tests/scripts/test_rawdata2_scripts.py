from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


RETAINED_SUITES = (
    "run_oracle_suite.sh",
)

RETAINED_WRAPPERS = (
    "run_oracle_gpu0_batch.sh",
    "run_oracle_gpu1_batch.sh",
    "run_oracle_gpu0_fast.sh",
    "run_oracle_gpu1_fast.sh",
    "run_oracle_gpu0_batch_adam.sh",
    "run_oracle_gpu1_batch_adam.sh",
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
    ega_pos = content.index('local ega_run_name="ega_${mode}_uupdate_dmodel_${RUN_TAG}"')
    topk_pos = content.index('"topk_${mode}_uupdate_dmodel_${RUN_TAG}"')
    assert fedavg_pos < ega_pos < topk_pos
    assert '--algorithms fedavg,topk,ega' in content
    for marker in (
        'RUN_TAG="${RUN_TAG:-oracle_attackfreq5}"',
        'TRACKING_TAG="${TRACKING_TAG:-oracle-attackfreq5}"',
        'ATTACK_ENABLED="${ATTACK_ENABLED:-true}"',
        'ATTACK_FREQUENCY_ROUNDS="${ATTACK_FREQUENCY_ROUNDS:-5}"',
        'ATTACK_MAX_SAMPLES="${ATTACK_MAX_SAMPLES:-}"',
        'LOSS_NAME="${LOSS_NAME:-mse}"',
        'LOSS_TAG="${LOSS_TAG:-${LOSS_NAME}}"',
        'TRAIN_OPTIMIZER="${TRAIN_OPTIMIZER:-}"',
        'TRAIN_MOMENTUM="${TRAIN_MOMENTUM:-}"',
        'TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-}"',
        'TRAIN_OPTIMIZER_EPS="${TRAIN_OPTIMIZER_EPS:-}"',
        'EVAL_MODE="${EVAL_MODE:-protocol}"',
        'SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-true}"',
        'MODEL_DROPOUT="${MODEL_DROPOUT:-0.1}"',
        'runtime.num_threads=1',
        'runtime.num_interop_threads=1',
        'FEDERATED_ALGORITHMS="${FEDERATED_ALGORITHMS:-fedavg,topk,ega}"',
        'TOPK_FRACTION="${TOPK_FRACTION:-}"',
        'QSGD_LEVELS="${QSGD_LEVELS:-}"',
        'RANDOMK_FRACTION="${RANDOMK_FRACTION:-}"',
        'QINT8_DTYPE="${QINT8_DTYPE:-}"',
        'EGA_DOWNLOAD_METHOD="${EGA_DOWNLOAD_METHOD:-}"',
        'EGA_PRETRAIN_DEVICE="${EGA_PRETRAIN_DEVICE:-}"',
        'emit_optional_override() {',
        'attack.enabled=%s',
        'attack.max_samples=%s',
        'training.loss=%s',
        'training.optimizer=%s',
        'training.momentum=%s',
        'training.weight_decay=%s',
        'training.optimizer_eps=%s',
        'evaluation.mode=%s',
        'data.shuffle_train=%s',
        'model.dropout=%s',
        'federated.topk_fraction',
        'federated.qsgd_levels',
        'federated.quantization_dtype',
        'ega.download_method',
        'TRAIN_OPTIMIZER=adam TRAIN_LR=0.001 bash SCRIPT --modes centralized,single_sync',
        'EVAL_MODE=protocol SHUFFLE_TRAIN=true MODEL_DROPOUT=0.1 bash SCRIPT --modes single_sync',
        'FEDERATED_ALGORITHMS=fedavg,ega bash SCRIPT --modes single_sync',
        'RUNTIME_DEVICE=cuda:1 bash SCRIPT --modes single_sync',
        'EGA_DOWNLOAD_METHOD=dense EGA_PRETRAIN_DEVICE=cuda:1 bash SCRIPT --modes single_sync',
    ):
        assert marker in content
    assert 'ega.download_method=ega' not in content
    assert 'topk10-' not in content
    assert 'qsgd63-' not in content
    assert 'ega.artifact_path' in content
    assert 'rm -f "${ega_artifact}"' not in content


def test_oracle_gpu_wrappers_call_generic_suite_without_duplicate_attackfreq5_runs():
    for name in RETAINED_WRAPPERS:
        content = _assert_executable(name)
        assert "run_oracle_suite.sh" in content
        assert "run_oracle_suite_param.sh" not in content
        assert "_mae.sh" not in content
        assert "LOSS_NAME=mse LOSS_TAG=mse" in content or "LOSS_NAME=mae LOSS_TAG=mae" in content

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


def test_removed_legacy_scripts_are_absent():
    for name in REMOVED_LEGACY:
        assert not (SCRIPT_DIR / name).exists(), name
