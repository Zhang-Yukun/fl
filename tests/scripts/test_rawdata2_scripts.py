from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


def test_rawdata2_run_scripts_are_separate_and_executable():
    """Default example runs expose separate bash entry points."""

    expected = {
        "run_rawdata2_centralized.sh": ["--mode centralized", "outputs/centralized"],
        "run_rawdata2_fedavg.sh": ["--mode federated", "federated.algorithm=fedavg"],
        "run_rawdata2_soteriafl.sh": ["configs/soteriafl.yaml"],
        "run_rawdata2_dp_topk.sh": ["configs/dp_topk.yaml"],
        "run_rawdata2_secure_quantized_fedavg.sh": ["configs/secure_quantized_fedavg.yaml"],
        "run_rawdata2_fedaware.sh": ["configs/fedaware.yaml"],
        "run_rawdata2_fedlab_topk.sh": ["configs/topk.yaml"],
        "run_rawdata2_randomk.sh": ["configs/randomk.yaml"],
        "run_rawdata2_sign.sh": ["configs/sign.yaml"],
        "run_rawdata2_qsgd.sh": ["configs/qsgd.yaml"],
        "run_rawdata2_ega.sh": ["configs/ega.yaml"],
    }
    for name, markers in expected.items():
        path = SCRIPT_DIR / name
        assert path.exists()
        assert path.stat().st_mode & 0o111
        content = path.read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash")
        for marker in markers:
            assert marker in content


def test_rawdata2_all_script_runs_each_entrypoint():
    """The aggregate example script invokes the default three experiment scripts."""

    path = SCRIPT_DIR / "run_rawdata2_all.sh"
    assert path.exists()
    assert path.stat().st_mode & 0o111
    content = path.read_text(encoding="utf-8")
    for script_name in (
        "scripts/run_rawdata2_centralized.sh",
        "scripts/run_rawdata2_fedavg.sh",
        "scripts/run_rawdata2_dp_topk.sh",
    ):
        assert script_name in content
    assert 'bash "${run_script}" "$@"' in content


def test_formal_suite_script_exists_and_lists_all_requested_runs():
    """The formal suite script should run the requested nine experiments sequentially."""

    path = SCRIPT_DIR / "run_formal_centralized_fedavg_qint8_suite.sh"
    assert path.exists()
    assert path.stat().st_mode & 0o111
    content = path.read_text(encoding="utf-8")
    for marker in (
        "centralized",
        "fedavg_single_sync",
        "fedavg_single_async",
        "fedavg_grpc_sync",
        "fedavg_grpc_async",
        "qint8_single_sync",
        "qint8_single_async",
        "qint8_grpc_sync",
        "qint8_grpc_async",
        "configs/fedavg.yaml",
        "configs/secure_quantized_fedavg.yaml",
        "federated.quantization_dtype=int8",
        "-m fedlab.entrypoints.train",
        "-m fedlab.entrypoints.server",
        "-m fedlab.entrypoints.client",
    ):
        assert marker in content


def test_oracle_suite_runs_ega_after_fedavg_and_uses_env_parameters():
    """The generic oracle suite should derive scenario behavior from env vars and keep EGA right after FedAvg."""

    path = SCRIPT_DIR / "run_oracle_suite.sh"
    assert path.exists()
    assert path.stat().st_mode & 0o111
    content = path.read_text(encoding="utf-8")
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
        'SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-false}"',
        'MODEL_DROPOUT="${MODEL_DROPOUT:-0.0}"',
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
        'EGA_DOWNLOAD_METHOD=dense EGA_PRETRAIN_DEVICE=cuda:1 bash SCRIPT --modes single_sync',
    ):
        assert marker in content
    assert 'ega.download_method=ega' not in content
    assert 'topk10-' not in content
    assert 'qsgd63-' not in content
    assert 'ega.artifact_path' in content
    assert 'rm -f "${ega_artifact}"' not in content


def test_oracle_gpu_wrappers_call_generic_suite_without_duplicate_attackfreq5_runs():
    wrappers = (
        "run_oracle_gpu0_batch.sh",
        "run_oracle_gpu1_batch.sh",
        "run_oracle_gpu0_fast.sh",
        "run_oracle_gpu1_fast.sh",
        "run_oracle_gpu0_batch_adam.sh",
        "run_oracle_gpu1_batch_adam.sh",
    )
    for name in wrappers:
        content = (SCRIPT_DIR / name).read_text(encoding="utf-8")
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

    for removed_name in (
        "run_oracle_attackfreq1_suite.sh",
        "run_oracle_noattack_suite.sh",
        "run_oracle_attackfreq5_1000r_pat100_suite.sh",
        "run_oracle_attackfreq5_maxsamples8_suite.sh",
        "run_oracle_attackfreq1_suite_mae.sh",
        "run_oracle_noattack_suite_mae.sh",
        "run_oracle_attackfreq5_suite_mae.sh",
        "run_oracle_attackfreq5_1000r_pat100_suite_mae.sh",
        "run_oracle_attackfreq5_maxsamples8_suite_mae.sh",
    ):
        assert not (SCRIPT_DIR / removed_name).exists()
