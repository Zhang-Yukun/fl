#!/usr/bin/env python3
"""Plot validation and communication curves for a list of experiment folders.

Example:
    python -m fedlab.tools.plot_experiment_suite         outputs/formal_suite/centralized         outputs/formal_suite/fedavg_single_sync         outputs/formal_suite/qint8_single_sync         --output-dir outputs/formal_suite/plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


LOWER_IS_BETTER_METRICS = ("loss", "mse", "mae", "mape")
VALIDATION_LOSS_CANDIDATES = (
    ("val_loss", "Validation Loss"),
    ("active_val_loss", "Validation Loss"),
    ("protocol_val_loss", "Validation Loss"),
    ("val_mse", "Validation MSE"),
    ("active_val_mse", "Validation MSE"),
    ("protocol_val_mse", "Validation MSE"),
    ("val_metrics.mse", "Validation MSE"),
    ("active_val_metrics.mse", "Validation MSE"),
    ("protocol_val_metrics.mse", "Validation MSE"),
)


@dataclass
class RunSeries:
    """One run summarized for plotting and tabular reporting."""

    label: str
    run_dir: Path
    kind: str
    algorithm: str
    steps: list[int]
    val_mse: list[float]
    val_mae: list[float]
    val_mape: list[float]
    cumulative_communication_bytes: list[int]
    final_test_mse: float | None
    final_test_mape: float | None
    final_total_communication_bytes: int | None
    validation_loss_label: str | None = None
    validation_loss: list[float] = field(default_factory=list)
    cumulative_upload_bytes: list[int] = field(default_factory=list)
    final_total_upload_bytes: int | None = None


def load_json(path: Path) -> Any:
    """Load one JSON artifact file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_record_value(record: dict[str, Any], dotted_key: str) -> Any:
    current: Any = record
    for part in dotted_key.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _load_metric_series(history: list[dict[str, Any]], key: str) -> list[float]:
    """Load one metric series only when every point includes the metric."""

    if not history:
        return []
    values: list[float] = []
    for record in history:
        value = _resolve_record_value(record, key)
        if value is None:
            return []
        values.append(float(value))
    return values


def _load_validation_loss_series(history: list[dict[str, Any]]) -> tuple[str | None, list[float]]:
    """Resolve one validation-loss-like series from common artifact fields."""

    for key, label in VALIDATION_LOSS_CANDIDATES:
        values = _load_metric_series(history, key)
        if values:
            return label, values
    primary_names = [str(record.get("primary_metric_name", "")).lower() for record in history if record.get("primary_metric_name") is not None]
    if history and primary_names and len(set(primary_names)) == 1 and primary_names[0] in LOWER_IS_BETTER_METRICS:
        values = _load_metric_series(history, "primary_metric_value")
        if values:
            metric_name = primary_names[0]
            metric_label = "Validation Loss" if metric_name == "loss" else f"Validation {metric_name.upper()}"
            return metric_label, values
    return None, []


def _load_centralized_series(run_dir: Path, metrics: dict[str, Any], summary: dict[str, Any] | None) -> RunSeries:
    """Load one centralized experiment artifact directory."""

    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Centralized metrics history is empty in {run_dir}")
    steps = [int(record["round"]) for record in history]
    test = (summary or {}).get("test") or {}
    test_mse = float(test["mse"]) if "mse" in test else None
    test_mape = float(test["mape"]) if "mape" in test else None
    algorithm = str((summary or {}).get("algorithm", "centralized"))
    validation_loss_label, validation_loss = _load_validation_loss_series(history)
    return RunSeries(
        label=run_dir.name,
        run_dir=run_dir,
        kind="centralized",
        algorithm=algorithm,
        steps=steps,
        val_mse=_load_metric_series(history, "val_mse"),
        val_mae=_load_metric_series(history, "val_mae"),
        val_mape=_load_metric_series(history, "val_mape"),
        cumulative_communication_bytes=[],
        final_test_mse=test_mse,
        final_test_mape=test_mape,
        final_total_communication_bytes=None,
        validation_loss_label=validation_loss_label,
        validation_loss=validation_loss,
        cumulative_upload_bytes=[],
        final_total_upload_bytes=None,
    )


def _resolve_round_communication_bytes(record: dict[str, Any], communication_mode: str) -> int:
    """Resolve one round communication volume from old or new artifact fields."""

    if communication_mode == "parameter":
        direct = record.get("total_parameter_bytes")
        if direct is not None:
            return int(direct)
        upload = int(record.get("total_parameter_upload_bytes", record.get("total_upload_bytes", 0)))
        download = int(record.get("total_parameter_download_bytes", record.get("total_download_bytes", 0)))
        return upload + download
    direct = record.get("total_transport_bytes")
    if direct is not None:
        return int(direct)
    upload = int(record.get("total_transport_upload_bytes", record.get("total_parameter_upload_bytes", record.get("total_upload_bytes", 0))))
    download = int(record.get("total_transport_download_bytes", record.get("total_parameter_download_bytes", record.get("total_download_bytes", 0))))
    return upload + download


def _resolve_round_upload_bytes(record: dict[str, Any], communication_mode: str) -> int:
    """Resolve one round upload volume from old or new artifact fields."""

    if communication_mode == "parameter":
        return int(record.get("total_parameter_upload_bytes", record.get("total_upload_bytes", 0)))
    return int(record.get("total_transport_upload_bytes", record.get("total_parameter_upload_bytes", record.get("total_upload_bytes", 0))))


def _resolve_summary_total_bytes(summary: dict[str, Any] | None, communication_mode: str) -> int | None:
    """Resolve final total communication bytes from summary when available."""

    if summary is None:
        return None
    if communication_mode == "parameter":
        if summary.get("total_parameter_bytes") is not None:
            return int(summary["total_parameter_bytes"])
        upload = summary.get("total_parameter_upload_bytes", summary.get("total_upload_bytes"))
        download = summary.get("total_parameter_download_bytes", summary.get("total_download_bytes"))
        if upload is not None and download is not None:
            return int(upload) + int(download)
        return None
    if summary.get("total_transport_bytes") is not None:
        return int(summary["total_transport_bytes"])
    upload = summary.get("total_transport_upload_bytes", summary.get("total_parameter_upload_bytes", summary.get("total_upload_bytes")))
    download = summary.get("total_transport_download_bytes", summary.get("total_parameter_download_bytes", summary.get("total_download_bytes")))
    if upload is not None and download is not None:
        return int(upload) + int(download)
    return None


def _resolve_summary_total_upload_bytes(summary: dict[str, Any] | None, communication_mode: str) -> int | None:
    """Resolve final cumulative upload bytes from summary when available."""

    if summary is None:
        return None
    if communication_mode == "parameter":
        value = summary.get("total_parameter_upload_bytes", summary.get("total_upload_bytes"))
    else:
        value = summary.get("total_transport_upload_bytes", summary.get("total_parameter_upload_bytes", summary.get("total_upload_bytes")))
    return None if value is None else int(value)


def _load_federated_series(
    run_dir: Path,
    metrics: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    communication_mode: str,
) -> RunSeries:
    """Load one federated experiment artifact directory."""

    if not metrics:
        raise ValueError(f"Federated metrics history is empty in {run_dir}")
    steps = [int(record.get("round", index)) for index, record in enumerate(metrics)]
    cumulative: list[int] = []
    cumulative_upload: list[int] = []
    running_total = 0
    running_upload = 0
    for record in metrics:
        running_total += _resolve_round_communication_bytes(record, communication_mode)
        cumulative.append(running_total)
        running_upload += _resolve_round_upload_bytes(record, communication_mode)
        cumulative_upload.append(running_upload)
    test = (summary or {}).get("test") or {}
    test_mse = float(test["mse"]) if "mse" in test else None
    test_mape = float(test["mape"]) if "mape" in test else None
    total_from_summary = _resolve_summary_total_bytes(summary, communication_mode)
    final_total = int(total_from_summary) if total_from_summary is not None else (cumulative[-1] if cumulative else 0)
    total_upload_from_summary = _resolve_summary_total_upload_bytes(summary, communication_mode)
    final_total_upload = int(total_upload_from_summary) if total_upload_from_summary is not None else (cumulative_upload[-1] if cumulative_upload else 0)
    algorithm = str(metrics[-1].get("algorithm", "unknown"))
    validation_loss_label, validation_loss = _load_validation_loss_series(metrics)
    return RunSeries(
        label=run_dir.name,
        run_dir=run_dir,
        kind="federated",
        algorithm=algorithm,
        steps=steps,
        val_mse=_load_metric_series(metrics, "val_mse"),
        val_mae=_load_metric_series(metrics, "val_mae"),
        val_mape=_load_metric_series(metrics, "val_mape"),
        cumulative_communication_bytes=cumulative,
        final_test_mse=test_mse,
        final_test_mape=test_mape,
        final_total_communication_bytes=final_total,
        validation_loss_label=validation_loss_label,
        validation_loss=validation_loss,
        cumulative_upload_bytes=cumulative_upload,
        final_total_upload_bytes=final_total_upload,
    )


def load_run_series(run_dir: Path) -> RunSeries:
    """Load one experiment folder into a plotting/reporting series.

    The tool accepts both centralized artifacts (``metrics.json`` stores a
    ``history`` object) and federated artifacts (``metrics.json`` stores a list
    of round records).
    """

    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
    metrics = load_json(metrics_path)
    summary = load_json(summary_path) if summary_path.exists() else None
    if isinstance(metrics, dict):
        return _load_centralized_series(run_dir, metrics, summary)
    if isinstance(metrics, list):
        return _load_federated_series(run_dir, metrics, summary, "parameter")
    raise TypeError(f"Unsupported metrics.json format in {run_dir}: {type(metrics)!r}")


def _series_colors(count: int) -> list[str]:
    """Return deterministic series colors matching matplotlib's default line cycle."""

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return [palette[index % len(palette)] for index in range(max(0, int(count)))]


def format_bytes(num_bytes: int | float | None) -> str:
    """Format bytes into a readable unit string."""

    if num_bytes is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def _format_byte_axis_value(value: float, _position: float) -> str:
    """Format one axis tick using readable byte units."""

    return format_bytes(value)


def _apply_byte_axis_formatter() -> None:
    """Apply readable byte-unit formatting to the current y axis."""

    plt.gca().yaxis.set_major_formatter(FuncFormatter(_format_byte_axis_value))


def _plot_series_with_optional_padding(
    steps: list[int],
    values: list[float],
    label: str,
    pad_to_step: int | None,
) -> None:
    """Plot one series and optionally extend it with a dashed flat tail."""

    line, = plt.plot(steps, values, linewidth=1.6, label=label)
    if pad_to_step is None or not steps or steps[-1] >= pad_to_step:
        return
    padded_steps = list(range(steps[-1] + 1, pad_to_step + 1))
    if not padded_steps:
        return
    plt.plot(
        [steps[-1], *padded_steps],
        [values[-1], *([values[-1]] * len(padded_steps))],
        linestyle="--",
        linewidth=1.2,
        color=line.get_color(),
        alpha=0.9,
    )


def plot_validation_metric(
    series_list: list[RunSeries],
    output_path: Path,
    metric_name: str,
    pad_to_max_step: bool = False,
) -> Path:
    """Plot one validation metric against epoch or round."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    max_step = max((series.steps[-1] for series in series_list if series.steps), default=None) if pad_to_max_step else None
    for series in series_list:
        values = getattr(series, f"val_{metric_name}")
        if not series.steps or not values:
            continue
        x_label = "round"
        _plot_series_with_optional_padding(series.steps, values, f"{series.label} ({x_label})", max_step)
    plt.xlabel("Epoch / Round")
    plt.ylabel(f"Validation {metric_name.upper()}")
    plt.title(f"Validation {metric_name.upper()} vs Epoch / Round")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_validation_mse(series_list: list[RunSeries], output_path: Path, pad_to_max_step: bool = False) -> Path:
    """Plot validation MSE against epoch or round."""

    return plot_validation_metric(series_list, output_path, "mse", pad_to_max_step)


def plot_validation_mae(series_list: list[RunSeries], output_path: Path, pad_to_max_step: bool = False) -> Path:
    """Plot validation MAE against epoch or round."""

    return plot_validation_metric(series_list, output_path, "mae", pad_to_max_step)


def plot_validation_mape(series_list: list[RunSeries], output_path: Path, pad_to_max_step: bool = False) -> Path:
    """Plot validation MAPE against epoch or round."""

    return plot_validation_metric(series_list, output_path, "mape", pad_to_max_step)


def plot_validation_loss(series_list: list[RunSeries], output_path: Path, pad_to_max_step: bool = False) -> Path:
    """Plot one validation-loss-like curve for the selected runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    max_step = max((series.steps[-1] for series in series_list if series.steps), default=None) if pad_to_max_step else None
    plotted = False
    ylabels = {series.validation_loss_label for series in series_list if series.validation_loss and series.validation_loss_label}
    for series in series_list:
        if not series.steps or not series.validation_loss:
            continue
        _plot_series_with_optional_padding(series.steps, series.validation_loss, series.label, max_step)
        plotted = True
    if not plotted:
        raise ValueError("No runs with validation loss-like history were provided")
    ylabel = next(iter(ylabels)) if len(ylabels) == 1 else "Validation Metric"
    plt.xlabel("Epoch / Round")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Epoch / Round")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_cumulative_communication(series_list: list[RunSeries], output_path: Path, communication_mode: str) -> Path:
    """Plot cumulative communication bytes against round for federated runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted = False
    for series in series_list:
        if series.kind != "federated" or not series.cumulative_communication_bytes:
            continue
        plt.plot(series.steps, series.cumulative_communication_bytes, linewidth=1.6, label=series.label)
        plotted = True
    if not plotted:
        raise ValueError("No federated runs with communication history were provided")
    ylabel = "Cumulative Parameter Communication (bytes)" if communication_mode == "parameter" else "Cumulative Transport Communication (bytes)"
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(f"Cumulative {communication_mode.capitalize()} Communication vs Round")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_cumulative_parameter_upload(series_list: list[RunSeries], output_path: Path) -> Path:
    """Plot cumulative parameter upload bytes against round for federated runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted = False
    for series in series_list:
        if series.kind != "federated" or not series.cumulative_upload_bytes:
            continue
        plt.plot(series.steps, series.cumulative_upload_bytes, linewidth=1.6, label=series.label)
        plotted = True
    if not plotted:
        raise ValueError("No federated runs with parameter upload history were provided")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Parameter Upload")
    plt.title("Cumulative Parameter Upload vs Round")
    _apply_byte_axis_formatter()
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_total_communication_bars(series_list: list[RunSeries], output_path: Path, communication_mode: str) -> Path:
    """Plot final total communication bytes for federated runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    federated = [series for series in series_list if series.kind == "federated" and series.final_total_communication_bytes is not None]
    if not federated:
        raise ValueError("No federated runs with final communication totals were provided")
    labels = [series.label for series in federated]
    values = [int(series.final_total_communication_bytes or 0) for series in federated]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color=_series_colors(len(federated)))
    ylabel = "Final Total Parameter Communication (bytes)" if communication_mode == "parameter" else "Final Total Transport Communication (bytes)"
    plt.ylabel(ylabel)
    plt.title(f"Final Total {communication_mode.capitalize()} Communication")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_total_parameter_upload_bars(series_list: list[RunSeries], output_path: Path) -> Path:
    """Plot final cumulative parameter upload bytes for federated runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    federated = [series for series in series_list if series.kind == "federated" and series.final_total_upload_bytes is not None]
    if not federated:
        raise ValueError("No federated runs with final parameter upload totals were provided")
    labels = [series.label for series in federated]
    values = [int(series.final_total_upload_bytes or 0) for series in federated]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color=_series_colors(len(federated)))
    plt.xlabel("Experiment / Algorithm")
    plt.ylabel("Total Parameter Upload")
    plt.title("Total Parameter Upload by Algorithm")
    _apply_byte_axis_formatter()
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_parameter_upload_compression_ratio_bars(series_list: list[RunSeries], output_path: Path) -> Path:
    """Plot FedAvg-relative parameter-upload compression ratios for federated runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    federated = [series for series in series_list if series.kind == "federated" and series.final_total_upload_bytes not in (None, 0)]
    if not federated:
        raise ValueError("No federated runs with final parameter upload totals were provided")
    if len(federated) < 1:
        raise ValueError("At least one federated run is required to compute upload compression ratios")
    fedavg_upload = federated[0].final_total_upload_bytes
    if fedavg_upload in (None, 0):
        raise ValueError("FedAvg parameter upload total is unavailable")
    labels = [series.label for series in federated]
    values = [float(fedavg_upload) / float(series.final_total_upload_bytes) for series in federated]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color=_series_colors(len(federated)))
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f}x",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.xlabel("Experiment / Algorithm")
    plt.ylabel("FedAvg Upload / Algorithm Upload")
    plt.title("Parameter Upload Compression Ratio by Algorithm")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def build_report_lines(series_list: list[RunSeries]) -> list[str]:
    """Build summary lines for stdout reporting using parameter-upload totals only."""

    lines = []
    fedavg_total = None
    if len(series_list) >= 2:
        fedavg_total = series_list[1].final_total_upload_bytes
    header = (
        f"{'label':<36} {'algorithm':<24} {'test_mse':>14} {'test_mape':>14} "
        f"{'total_upload':>18} {'vs_fedavg_upload':>18}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for index, series in enumerate(series_list):
        if index == 0:
            ratio_text = "n/a"
        elif index == 1:
            ratio_text = "1.000000"
        else:
            if fedavg_total is None or series.final_total_upload_bytes in (None, 0):
                ratio_text = "n/a"
            else:
                ratio_text = f"{float(fedavg_total) / float(series.final_total_upload_bytes):.6f}"
        test_mse_text = f"{series.final_test_mse:.6f}" if series.final_test_mse is not None else "n/a"
        test_mape_text = f"{series.final_test_mape:.6f}" if series.final_test_mape is not None else "n/a"
        lines.append(
            f"{series.label:<36} {series.algorithm:<24} {test_mse_text:>14} {test_mape_text:>14} "
            f"{format_bytes(series.final_total_upload_bytes):>18} {ratio_text:>18}"
        )
    return lines


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot validation curves and parameter-upload communication for several experiment folders. "
            "By convention, the first folder is centralized and the second is FedAvg."
        )
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Experiment artifact directories in display order")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory receiving plots and the text summary. Defaults to <first_run>/suite_plots",
    )
    parser.add_argument(
        "--pad-to-max-step",
        action="store_true",
        help="Extend shorter validation curves to the longest run with a dashed flat tail.",
    )
    return parser


def main() -> None:
    """Run the plotting/reporting CLI."""

    parser = build_argparser()
    args = parser.parse_args()
    run_dirs = [path.resolve() for path in args.run_dirs]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise NotADirectoryError(f"{run_dir} is not a directory")
    series_list = [load_run_series(run_dir) for run_dir in run_dirs]
    output_dir = args.output_dir.resolve() if args.output_dir is not None else run_dirs[0] / "suite_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []
    optional_plots = [
        (plot_validation_loss, (series_list, output_dir / "validation_loss_curve.png", args.pad_to_max_step)),
        (plot_validation_mse, (series_list, output_dir / "validation_mse_curve.png", args.pad_to_max_step)),
        (plot_validation_mae, (series_list, output_dir / "validation_mae_curve.png", args.pad_to_max_step)),
        (plot_validation_mape, (series_list, output_dir / "validation_mape_curve.png", args.pad_to_max_step)),
        (plot_cumulative_parameter_upload, (series_list, output_dir / "cumulative_parameter_upload_curve.png")),
        (plot_total_parameter_upload_bars, (series_list, output_dir / "total_parameter_upload_bar.png")),
        (plot_parameter_upload_compression_ratio_bars, (series_list, output_dir / "parameter_upload_compression_ratio_bar.png")),
    ]
    skipped: list[str] = []
    for function, fn_args in optional_plots:
        try:
            plot_paths.append(function(*fn_args))
        except ValueError as exc:
            skipped.append(f"{function.__name__}: {exc}")

    report_lines = build_report_lines(series_list)
    if skipped:
        report_lines.append("")
        report_lines.append("skipped_plots:")
        for item in skipped:
            report_lines.append(f"- {item}")
    report_path = output_dir / "suite_summary.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


    print(f"saved_plots: {[str(path) for path in plot_paths]}")
    print(f"saved_summary: {report_path}")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
