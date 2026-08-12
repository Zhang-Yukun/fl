import importlib.util
import json
import pytest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[2] / "fedlab" / "tools" / "compare_experiment_results.py"


spec = importlib.util.spec_from_file_location("compare_experiment_results", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_compare_experiment_results_summarizes_communication_and_performance(tmp_path):
    """The comparison helper aggregates total bytes and candidate performance gaps."""

    baseline_dir = tmp_path / "fedavg_run"
    candidate_dir = tmp_path / "candidate_run"
    baseline_dir.mkdir()
    candidate_dir.mkdir()

    baseline_metrics = [
        {
            "algorithm": "fedavg",
            "total_upload_bytes": 100,
            "total_download_bytes": 100,
            "fedavg_reference_upload_bytes": 100,
            "fedavg_reference_total_bytes": 200,
        },
        {
            "algorithm": "fedavg",
            "total_upload_bytes": 120,
            "total_download_bytes": 120,
            "fedavg_reference_upload_bytes": 120,
            "fedavg_reference_total_bytes": 240,
        },
    ]
    candidate_metrics = [
        {
            "algorithm": "secure_quantized_fedavg",
            "total_upload_bytes": 30,
            "total_download_bytes": 30,
            "fedavg_reference_upload_bytes": 120,
            "fedavg_reference_total_bytes": 240,
        },
        {
            "algorithm": "secure_quantized_fedavg",
            "total_upload_bytes": 40,
            "total_download_bytes": 40,
            "fedavg_reference_upload_bytes": 120,
            "fedavg_reference_total_bytes": 240,
        },
    ]
    baseline_summary = {"test": {"mse": 1.0, "mae": 2.0, "mape": 4.0}, "rounds": 2}
    candidate_summary = {"test": {"mse": 0.9, "mae": 2.2, "mape": 3.0}, "rounds": 2}

    _write_json(baseline_dir / "metrics.json", baseline_metrics)
    _write_json(candidate_dir / "metrics.json", candidate_metrics)
    _write_json(baseline_dir / "summary.json", baseline_summary)
    _write_json(candidate_dir / "summary.json", candidate_summary)
    (baseline_dir / "run.log").write_text("baseline", encoding="utf-8")
    (candidate_dir / "run.log").write_text("candidate", encoding="utf-8")

    baseline = module.summarize_run(baseline_dir)
    candidate = module.summarize_run(candidate_dir)
    communication = module.compute_communication_gaps(baseline, candidate)
    performance = module.compute_performance_gaps(baseline, candidate)
    shared = module.common_files(baseline_dir, candidate_dir, module.DEFAULT_COMPARE_FILES)

    assert baseline["total_parameter_bytes"] == 440
    assert baseline["total_transport_bytes"] == 440
    assert candidate["total_parameter_bytes"] == 140
    assert candidate["total_transport_bytes"] == 140
    assert communication["candidate_vs_baseline_parameter_ratio"] == 140 / 440
    assert communication["candidate_vs_baseline_transport_ratio"] == 140 / 440
    assert round(communication["candidate_parameter_reduction_percent"], 6) == round((1 - 140 / 440) * 100, 6)
    assert round(communication["candidate_transport_reduction_percent"], 6) == round((1 - 140 / 440) * 100, 6)
    assert performance["mse"]["absolute_delta"] == pytest.approx(-0.1)
    assert performance["mae"]["absolute_delta"] == pytest.approx(0.2)
    assert performance["mape"]["relative_percent"] == pytest.approx(-25.0)
    assert shared == ["summary.json", "metrics.json", "run.log"]


def test_compare_experiment_results_script_exists_and_is_executable():
    """The experiment comparison helper is exposed as an executable script."""

    assert SCRIPT_PATH.exists()
    assert SCRIPT_PATH.stat().st_mode & 0o111
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env python3")
    assert "Performance gaps (candidate - baseline, lower is better):" in content
    assert "candidate_vs_baseline_parameter_ratio" in content
    assert "candidate_vs_baseline_transport_ratio" in content
