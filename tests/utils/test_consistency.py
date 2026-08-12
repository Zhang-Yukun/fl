import json

from fedlab.utils.consistency import compare_fedavg_runs, load_run_artifacts


def _write_run(path, summary, metrics, attacks):
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (path / "attack_results.json").write_text(json.dumps(attacks), encoding="utf-8")


def test_load_run_artifacts_reads_expected_files(tmp_path):
    summary = {"rounds": 2}
    metrics = [{"round": 0}]
    attacks = [{"name": "DLG"}]
    _write_run(tmp_path, summary, metrics, attacks)

    loaded = load_run_artifacts(tmp_path)

    assert loaded["summary"] == summary
    assert loaded["metrics"] == metrics
    assert loaded["attack_results"] == attacks


def test_compare_fedavg_runs_ignores_timing_only_fields(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    summary_left = {"rounds": 2, "total_time_seconds": 1.0, "test": {"mse": 0.5}}
    summary_right = {"rounds": 2, "total_time_seconds": 3.0, "test": {"mse": 0.5}}
    metrics_left = [{"round": 0, "round_time_seconds": 0.2, "train_loss": 1.0}]
    metrics_right = [{"round": 0, "round_time_seconds": 0.6, "train_loss": 1.0}]
    attacks_left = [{"name": "DLG", "mse": 0.3, "time_seconds": 0.1}]
    attacks_right = [{"name": "DLG", "mse": 0.3, "time_seconds": 0.9}]
    _write_run(left, summary_left, metrics_left, attacks_left)
    _write_run(right, summary_right, metrics_right, attacks_right)

    assert compare_fedavg_runs(left, right) == []


def test_compare_fedavg_runs_reports_metric_mismatches(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, {"rounds": 2}, [{"train_loss": 1.0}], [{"name": "DLG", "mse": 0.3}])
    _write_run(right, {"rounds": 2}, [{"train_loss": 1.1}], [{"name": "DLG", "mse": 0.3}])

    diffs = compare_fedavg_runs(left, right)

    assert diffs
    assert any("train_loss" in diff for diff in diffs)


def test_compare_fedavg_runs_can_ignore_transport_fields(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, {"total_transport_bytes": 10, "total_parameter_bytes": 4}, [{"transport_download_bytes": 6, "parameter_download_bytes": 2}], [])
    _write_run(right, {"total_transport_bytes": 20, "total_parameter_bytes": 4}, [{"transport_download_bytes": 16, "parameter_download_bytes": 2}], [])

    assert compare_fedavg_runs(left, right, ignore_transport=True) == []
