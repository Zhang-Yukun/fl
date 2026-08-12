#!/usr/bin/env python3
"""Plot validation and communication curves for a list of experiment folders.

Example:
    python -m fedlab.tools.plot_experiment_suite         outputs/formal_suite/centralized         outputs/formal_suite/fedavg_single_sync         outputs/formal_suite/qint8_single_sync         --output-dir outputs/formal_suite/plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOWER_IS_BETTER_METRICS = ("mse", "mae", "mape")
COMMUNICATION_FIELDS = {
    "parameter": "parameter",
    "transport": "transport",
}


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
    final_total_communication_bytes: int | None


def load_json(path: Path) -> Any:
    """Load one JSON artifact file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_metric_series(history: list[dict[str, Any]], key: str) -> list[float]:
    """Load one metric series only when every point includes the metric."""

    if not history or any(key not in record for record in history):
        return []
    return [float(record[key]) for record in history]


def _load_centralized_series(run_dir: Path, metrics: dict[str, Any], summary: dict[str, Any] | None) -> RunSeries:
    """Load one centralized experiment artifact directory."""

    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Centralized metrics history is empty in {run_dir}")
    steps = [int(record["epoch"]) for record in history]
    test = (summary or {}).get("test") or {}
    test_mse = float(test["mse"]) if "mse" in test else None
    algorithm = str((summary or {}).get("algorithm", "centralized"))
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
        final_total_communication_bytes=None,
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
    running_total = 0
    for record in metrics:
        running_total += _resolve_round_communication_bytes(record, communication_mode)
        cumulative.append(running_total)
    test = (summary or {}).get("test") or {}
    test_mse = float(test["mse"]) if "mse" in test else None
    total_from_summary = _resolve_summary_total_bytes(summary, communication_mode)
    final_total = int(total_from_summary) if total_from_summary is not None else (cumulative[-1] if cumulative else 0)
    algorithm = str(metrics[-1].get("algorithm", "unknown"))
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
        final_total_communication_bytes=final_total,
    )


def load_run_series(run_dir: Path, communication_mode: str = "parameter") -> RunSeries:
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
    communication_field = COMMUNICATION_FIELDS[communication_mode]
    if isinstance(metrics, dict):
        return _load_centralized_series(run_dir, metrics, summary)
    if isinstance(metrics, list):
        return _load_federated_series(run_dir, metrics, summary, communication_field)
    raise TypeError(f"Unsupported metrics.json format in {run_dir}: {type(metrics)!r}")


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
        x_label = "epoch" if series.kind == "centralized" else "round"
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


def plot_total_communication_bars(series_list: list[RunSeries], output_path: Path, communication_mode: str) -> Path:
    """Plot final total communication bytes for federated runs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    federated = [series for series in series_list if series.kind == "federated" and series.final_total_communication_bytes is not None]
    if not federated:
        raise ValueError("No federated runs with final communication totals were provided")
    labels = [series.label for series in federated]
    values = [int(series.final_total_communication_bytes or 0) for series in federated]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    ylabel = "Final Total Parameter Communication (bytes)" if communication_mode == "parameter" else "Final Total Transport Communication (bytes)"
    plt.ylabel(ylabel)
    plt.title(f"Final Total {communication_mode.capitalize()} Communication")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def build_report_lines(series_list: list[RunSeries]) -> list[str]:
    """Build summary lines for stdout reporting.

    Convention:
    - The first folder is centralized.
    - The second folder is FedAvg baseline.
    - Later folders are candidate methods whose compression ratios are
      reported as ``fedavg_total / candidate_total``.
    """

    lines = []
    fedavg_total = None
    if len(series_list) >= 2:
        fedavg_total = series_list[1].final_total_communication_bytes
    header = (
        f"{'label':<36} {'algorithm':<24} {'test_mse':>14} "
        f"{'total_comm':>18} {'vs_fedavg_ratio':>18}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for index, series in enumerate(series_list):
        if index == 0:
            ratio_text = "n/a"
        elif index == 1:
            ratio_text = "1.000000"
        else:
            if fedavg_total is None or series.final_total_communication_bytes in (None, 0):
                ratio_text = "n/a"
            else:
                ratio_text = f"{float(fedavg_total) / float(series.final_total_communication_bytes):.6f}"
        test_mse_text = f"{series.final_test_mse:.6f}" if series.final_test_mse is not None else "n/a"
        lines.append(
            f"{series.label:<36} {series.algorithm:<24} {test_mse_text:>14} "
            f"{format_bytes(series.final_total_communication_bytes):>18} {ratio_text:>18}"
        )
    return lines


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot validation MSE and communication curves for several experiment folders. "
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
        "--communication-mode",
        choices=sorted(COMMUNICATION_FIELDS),
        default="parameter",
        help="Communication metric used in the plots and summary table.",
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
    series_list = [load_run_series(run_dir, args.communication_mode) for run_dir in run_dirs]
    output_dir = args.output_dir.resolve() if args.output_dir is not None else run_dirs[0] / "suite_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = [
        plot_validation_mse(series_list, output_dir / "validation_mse_curve.png", args.pad_to_max_step),
        plot_validation_mae(series_list, output_dir / "validation_mae_curve.png", args.pad_to_max_step),
        plot_validation_mape(series_list, output_dir / "validation_mape_curve.png", args.pad_to_max_step),
        plot_cumulative_communication(series_list, output_dir / f"cumulative_{args.communication_mode}_communication_curve.png", args.communication_mode),
        plot_total_communication_bars(series_list, output_dir / f"final_{args.communication_mode}_communication_bar.png", args.communication_mode),
    ]

    report_lines = build_report_lines(series_list)
    report_path = output_dir / "suite_summary.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"saved_plots: {[str(path) for path in plot_paths]}")
    print(f"saved_summary: {report_path}")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
