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
        "run_rawdata2_fedpetuning.sh": ["configs/rawdata2_fedpetuning.yaml"],
        "run_rawdata2_fedaware.sh": ["configs/rawdata2_fedaware.yaml"],
        "run_rawdata2_fedlab_topk.sh": ["configs/rawdata2_fedlab_topk.yaml"],
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


def test_rawdata2_fedaware_python_entrypoint_exists():
    """FedAWARE also provides a direct Python entrypoint for supplementary runs."""

    path = SCRIPT_DIR / "run_rawdata2_fedaware.py"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "configs/rawdata2_fedaware.yaml" in content
    assert 'config["federated"]["algorithm"] = "fedaware"' in content
