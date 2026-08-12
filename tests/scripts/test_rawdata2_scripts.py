from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


def test_rawdata2_run_scripts_are_separate_and_executable():
    """Rawdata2 default runs expose separate bash entry points."""

    expected = {
        "run_rawdata2_centralized.sh": ["--mode centralized", "rawdata2_patchtst_centralized"],
        "run_rawdata2_fedavg.sh": ["--mode federated", "federated.algorithm=fedavg"],
        "run_rawdata2_soteriafl.sh": ["configs/rawdata2_soteriafl.yaml"],
        "run_rawdata2_dp_topk.sh": ["configs/rawdata2_dp_topk.yaml"],
        "run_rawdata2_secure_quantized_fedavg.sh": ["configs/rawdata2_secure_quantized_fedavg.yaml"],
        "run_rawdata2_fedaware.sh": ["configs/rawdata2_fedaware.yaml"],
        "run_rawdata2_fedlab_topk.sh": ["configs/rawdata2_fedlab_topk.yaml"],
        "run_rawdata2_randomk.sh": ["configs/rawdata2_randomk.yaml"],
        "run_rawdata2_sign.sh": ["configs/rawdata2_sign.yaml"],
        "run_rawdata2_qsgd.sh": ["configs/rawdata2_qsgd.yaml"],
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
    """The aggregate rawdata2 script invokes the default three experiment scripts."""

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
        "configs/rawdata2_patchtst.yaml",
        "configs/rawdata2_secure_quantized_fedavg.yaml",
        "federated.quantization_dtype=int8",
        "-m federated_ts.entrypoints.train",
        "-m federated_ts.entrypoints.server",
        "-m federated_ts.entrypoints.client",
    ):
        assert marker in content
