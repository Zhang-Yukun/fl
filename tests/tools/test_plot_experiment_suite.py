import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[2] / "fedlab" / "tools" / "plot_experiment_suite.py"
spec = importlib.util.spec_from_file_location("plot_experiment_suite", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_plot_experiment_suite_loads_runs_plots_and_reports(tmp_path):
    """The suite plotting helper should handle centralized and federated runs."""

    centralized_dir = tmp_path / "centralized"
    fedavg_dir = tmp_path / "fedavg"
    topk_dir = tmp_path / "topk"
    for run_dir in (centralized_dir, fedavg_dir, topk_dir):
        run_dir.mkdir()

    _write_json(
        centralized_dir / "metrics.json",
        {
            "history": [
                {"round": 0, "val_mse": 1.0, "val_mae": 0.4, "val_mape": 10.0},
                {"round": 1, "val_mse": 0.8, "val_mae": 0.3, "val_mape": 8.0},
            ],
            "test": {"mse": 0.7},
            "rounds": 2,
        },
    )
    _write_json(centralized_dir / "summary.json", {"test": {"mse": 0.7, "mape": 7.0}, "rounds": 2})

    _write_json(
        fedavg_dir / "metrics.json",
        [
            {"round": 0, "algorithm": "fedavg", "val_mse": 1.1, "val_mae": 0.5, "val_mape": 11.0, "total_parameter_upload_bytes": 100, "total_parameter_bytes": 100, "total_transport_bytes": 120},
            {"round": 1, "algorithm": "fedavg", "val_mse": 0.9, "val_mae": 0.35, "val_mape": 9.0, "total_parameter_upload_bytes": 120, "total_parameter_bytes": 120, "total_transport_bytes": 140},
        ],
    )
    _write_json(
        fedavg_dir / "summary.json",
        {"test": {"mse": 0.85, "mape": 8.5}, "rounds": 2, "total_parameter_upload_bytes": 220, "total_parameter_bytes": 220, "total_transport_bytes": 260},
    )

    _write_json(
        topk_dir / "metrics.json",
        [
            {"round": 0, "algorithm": "sparse_fedavg", "val_mse": 1.2, "val_mae": 0.6, "val_mape": 12.0, "total_parameter_upload_bytes": 40, "total_parameter_bytes": 40, "total_transport_bytes": 60},
            {"round": 1, "algorithm": "sparse_fedavg", "val_mse": 0.95, "val_mae": 0.42, "val_mape": 9.5, "total_parameter_upload_bytes": 50, "total_parameter_bytes": 50, "total_transport_bytes": 70},
        ],
    )
    _write_json(
        topk_dir / "summary.json",
        {"test": {"mse": 0.9, "mape": 9.0}, "rounds": 2, "total_parameter_upload_bytes": 90, "total_parameter_bytes": 90, "total_transport_bytes": 130},
    )

    series = [
        module.load_run_series(centralized_dir),
        module.load_run_series(fedavg_dir),
        module.load_run_series(topk_dir),
    ]

    assert series[0].kind == "centralized"
    assert series[0].val_mae == [0.4, 0.3]
    assert series[0].val_mape == [10.0, 8.0]
    assert series[0].validation_loss_label == "Validation MSE"
    assert series[0].validation_loss == [1.0, 0.8]
    assert series[0].final_test_mape == 7.0
    assert series[1].kind == "federated"
    assert series[1].cumulative_upload_bytes == [100, 220]
    assert series[1].final_total_upload_bytes == 220
    assert series[2].cumulative_upload_bytes == [40, 90]

    output_dir = tmp_path / "plots"
    loss_plot = module.plot_validation_loss(series, output_dir / "loss.png")
    mse_plot = module.plot_validation_mse(series, output_dir / "mse.png")
    mae_plot = module.plot_validation_mae(series, output_dir / "mae.png")
    mape_plot = module.plot_validation_mape(series, output_dir / "mape.png")
    upload_curve = module.plot_cumulative_parameter_upload(series, output_dir / "upload_curve.png")
    upload_bar = module.plot_total_parameter_upload_bars(series, output_dir / "upload_bar.png")
    ratio_bar = module.plot_parameter_upload_compression_ratio_bars(series, output_dir / "ratio_bar.png")
    report_lines = module.build_report_lines(series)

    assert loss_plot.exists()
    assert mse_plot.exists()
    assert mae_plot.exists()
    assert mape_plot.exists()
    assert upload_curve.exists()
    assert upload_bar.exists()
    assert ratio_bar.exists()
    assert any("fedavg" in line for line in report_lines)
    assert any("8.500000" in line for line in report_lines)
    assert any("2.444444" in line for line in report_lines)


def test_plot_experiment_suite_script_exists_and_is_executable():
    """The experiment suite plotting helper should be exposed as an executable script."""

    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env python3")
    assert "validation_loss_curve.png" in content
    assert "cumulative_parameter_upload_curve.png" in content
    assert "total_parameter_upload_bar.png" in content
    assert "parameter_upload_compression_ratio_bar.png" in content
    assert "validation_mse_curve.png" in content
    assert "vs_fedavg_upload" in content
    assert "--pad-to-max-step" in content


def test_plot_experiment_suite_supports_round_upload_download_fields(tmp_path):
    """The plotting helper should recover communication totals from upload/download fields."""

    fedavg_dir = tmp_path / "fedavg"
    qint8_dir = tmp_path / "qint8"
    fedavg_dir.mkdir()
    qint8_dir.mkdir()

    _write_json(
        fedavg_dir / "metrics.json",
        [
            {"round": 0, "algorithm": "fedavg", "val_mse": 1.0, "val_mae": 0.4, "val_mape": 10.0, "total_upload_bytes": 100, "total_download_bytes": 120},
            {"round": 1, "algorithm": "fedavg", "val_mse": 0.9, "val_mae": 0.3, "val_mape": 9.0, "total_upload_bytes": 110, "total_download_bytes": 130},
        ],
    )
    _write_json(fedavg_dir / "summary.json", {"test": {"mse": 0.8, "mape": 8.0}, "rounds": 2})

    _write_json(
        qint8_dir / "metrics.json",
        [
            {"round": 0, "algorithm": "secure_quantized_fedavg", "val_mse": 1.1, "val_mae": 0.45, "val_mape": 10.5, "total_upload_bytes": 30, "total_download_bytes": 40},
            {"round": 1, "algorithm": "secure_quantized_fedavg", "val_mse": 0.95, "val_mae": 0.32, "val_mape": 9.3, "total_upload_bytes": 35, "total_download_bytes": 45},
        ],
    )
    _write_json(qint8_dir / "summary.json", {"test": {"mse": 0.85, "mape": 8.5}, "rounds": 2})

    fedavg = module.load_run_series(fedavg_dir)
    qint8 = module.load_run_series(qint8_dir)

    assert fedavg.cumulative_upload_bytes == [100, 210]
    assert fedavg.final_total_upload_bytes == 210
    assert qint8.cumulative_upload_bytes == [30, 65]
    assert qint8.final_total_upload_bytes == 65

    report_lines = module.build_report_lines([
        module.RunSeries("centralized", tmp_path, "centralized", "centralized", [0], [1.0], [0.4], [10.0], [], 0.7, 7.0, None),
        fedavg,
        qint8,
    ])
    assert any("3.230769" in line for line in report_lines)


def test_plot_experiment_suite_padding_keeps_shorter_runs_visible(tmp_path):
    """Shorter runs should still plot with a dashed tail when padding is enabled."""

    short = module.RunSeries(
        label="short",
        run_dir=tmp_path,
        kind="federated",
        algorithm="fedavg",
        steps=[0, 1],
        val_mse=[1.0, 0.9],
        val_mae=[0.4, 0.3],
        val_mape=[10.0, 9.0],
        cumulative_communication_bytes=[],
        final_test_mse=0.8,
        final_test_mape=8.0,
        final_total_communication_bytes=None,
        validation_loss_label="Validation MSE",
        validation_loss=[1.0, 0.9],
        cumulative_upload_bytes=[6, 12],
        final_total_upload_bytes=12,
    )
    long = module.RunSeries(
        label="long",
        run_dir=tmp_path,
        kind="federated",
        algorithm="qint8",
        steps=[0, 1, 2, 3],
        val_mse=[1.2, 1.1, 1.0, 0.95],
        val_mae=[0.5, 0.45, 0.4, 0.35],
        val_mape=[12.0, 11.0, 10.0, 9.5],
        cumulative_communication_bytes=[],
        final_test_mse=0.9,
        final_test_mape=9.0,
        final_total_communication_bytes=None,
        validation_loss_label="Validation MSE",
        validation_loss=[1.2, 1.1, 1.0, 0.95],
        cumulative_upload_bytes=[3, 6, 9, 12],
        final_total_upload_bytes=12,
    )

    output_path = tmp_path / "pad_mse.png"
    result = module.plot_validation_loss([short, long], output_path, pad_to_max_step=True)

    assert result.exists()


def test_plot_experiment_suite_prefers_validation_loss_keys_when_present(tmp_path):
    run_dir = tmp_path / 'ega_like'
    run_dir.mkdir()
    _write_json(
        run_dir / 'metrics.json',
        [
            {'round': 0, 'algorithm': 'ega_fedavg', 'protocol_val_mse': 0.8, 'total_parameter_upload_bytes': 20},
            {'round': 1, 'algorithm': 'ega_fedavg', 'protocol_val_mse': 0.6, 'total_parameter_upload_bytes': 25},
        ],
    )
    _write_json(run_dir / 'summary.json', {'total_parameter_upload_bytes': 45})

    series = module.load_run_series(run_dir)

    assert series.validation_loss_label == 'Validation MSE'
    assert series.validation_loss == [0.8, 0.6]
    assert series.cumulative_upload_bytes == [20, 45]
    assert series.final_total_upload_bytes == 45


def test_plot_experiment_suite_bar_colors_follow_default_line_cycle():
    colors = module._series_colors(4)
    palette = module.plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert colors == palette[:4]


def test_plot_experiment_suite_upload_ratio_bar_uses_fedavg_baseline(tmp_path):
    fedavg = module.RunSeries(
        label="fedavg",
        run_dir=tmp_path,
        kind="federated",
        algorithm="fedavg",
        steps=[0, 1],
        val_mse=[1.0, 0.9],
        val_mae=[0.4, 0.3],
        val_mape=[10.0, 9.0],
        cumulative_communication_bytes=[],
        final_test_mse=0.8,
        final_test_mape=8.0,
        final_total_communication_bytes=None,
        validation_loss_label="Validation MSE",
        validation_loss=[1.0, 0.9],
        cumulative_upload_bytes=[100, 220],
        final_total_upload_bytes=220,
    )
    topk = module.RunSeries(
        label="topk",
        run_dir=tmp_path,
        kind="federated",
        algorithm="sparse_fedavg",
        steps=[0, 1],
        val_mse=[1.1, 0.95],
        val_mae=[0.45, 0.35],
        val_mape=[10.5, 9.5],
        cumulative_communication_bytes=[],
        final_test_mse=0.85,
        final_test_mape=8.5,
        final_total_communication_bytes=None,
        validation_loss_label="Validation MSE",
        validation_loss=[1.1, 0.95],
        cumulative_upload_bytes=[40, 90],
        final_total_upload_bytes=90,
    )

    output_path = tmp_path / "ratio_bar.png"
    result = module.plot_parameter_upload_compression_ratio_bars([fedavg, topk], output_path)

    assert result.exists()


def test_plot_experiment_suite_formats_byte_axis_ticks():
    assert module._format_byte_axis_value(1024 ** 2, 0) == "1.00MB"
    assert module._format_byte_axis_value(3 * 1024 ** 3, 0) == "3.00GB"
