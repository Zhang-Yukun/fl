from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


def test_rawdata2_run_scripts_are_separate_and_executable():
    """Rawdata2 centralized, FedAvg, and Top-k runs have separate bash entry points."""

    expected = {
        "run_rawdata2_centralized.sh": ["--mode centralized", "rawdata2_patchtst_centralized"],
        "run_rawdata2_fedavg.sh": ["--mode federated", "federated.algorithm=fedavg"],
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
    """The aggregate rawdata2 script invokes the three concrete experiment scripts."""

    path = SCRIPT_DIR / "run_rawdata2_all.sh"
    assert path.exists()
    assert path.stat().st_mode & 0o111
    content = path.read_text(encoding="utf-8")
    for script_name in (
        "scripts/run_rawdata2_centralized.sh",
        "scripts/run_rawdata2_fedavg.sh",
        "scripts/run_rawdata2_fedlab_topk.sh",
    ):
        assert script_name in content
    assert 'bash "${run_script}" "$@"' in content
