#!/usr/bin/env python3
"""Plot saved attack reconstructions against the corresponding real series.

Example:
    python -m fedlab.tools.plot_attack_reconstructions         outputs/repro_pat50_attackviz_200/fedavg_seed2026_pat50_payloadv1         --output-dir outputs/repro_pat50_attackviz_200/fedavg_seed2026_pat50_payloadv1/attack_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_records(
    records: list[dict[str, Any]],
    *,
    sort_key: str,
    descending: bool,
    attack_name: str | None,
    client_id: str | None,
    round_index: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Filter and rank attack records before plotting."""

    filtered = records
    if attack_name is not None:
        filtered = [record for record in filtered if str(record.get("name")) == attack_name]
    if client_id is not None:
        filtered = [record for record in filtered if str(record.get("client_id")) == client_id]
    if round_index is not None:
        filtered = [record for record in filtered if int(record.get("round_index", -1)) == round_index]

    def key_fn(record: dict[str, Any]) -> tuple[int, float]:
        """Return a sortable key that keeps missing metrics at the end."""

        value = record.get(sort_key)
        if value is None:
            return (1, float("inf"))
        return (0, float(value))

    filtered = sorted(filtered, key=key_fn, reverse=descending)
    return filtered[:limit]




def should_plot_real_y(record: dict[str, Any], show_idlg_y: bool = False) -> bool:
    """Return whether real_y should appear on the plot for one attack record."""

    name = str(record.get("name", "")).lower()
    if name == "idlg" and not show_idlg_y:
        return False
    return True


def should_plot_reconstructed_y(record: dict[str, Any], show_idlg_y: bool = False) -> bool:
    """Return whether reconstructed_y should appear on the plot for one attack record."""

    name = str(record.get("name", "")).lower()
    if name == "idlg" and not show_idlg_y:
        return False
    return True

def _flatten_first_channel(tensor: torch.Tensor | None) -> list[float] | None:
    """Convert one saved tensor artifact into a 1D series for plotting."""

    if tensor is None:
        return None
    data = tensor.detach().cpu().float()
    if data.ndim == 3:
        return data[0, :, 0].tolist()
    if data.ndim == 2:
        return data[0, :].tolist()
    if data.ndim == 1:
        return data.tolist()
    raise ValueError(f"Unsupported tensor shape for plotting: {tuple(data.shape)}")


def plot_one_artifact(
    run_dir: Path,
    record: dict[str, Any],
    output_dir: Path,
    *,
    show_idlg_y: bool = False,
) -> Path:
    """Render one attack artifact into a PNG figure."""

    artifact_rel = record.get("artifact_path")
    if not artifact_rel:
        raise ValueError("Attack record does not include artifact_path")
    artifact = torch.load(run_dir / artifact_rel, map_location="cpu", weights_only=False)

    real_x = _flatten_first_channel(artifact.get("plot_reference_x") if artifact.get("plot_reference_x") is not None else (artifact.get("plot_real_x") if artifact.get("plot_real_x") is not None else (artifact.get("reference_x") if artifact.get("reference_x") is not None else artifact.get("real_x"))))
    recon_x = _flatten_first_channel(artifact.get("plot_reconstructed_x") if artifact.get("plot_reconstructed_x") is not None else artifact.get("reconstructed_x"))
    real_y = _flatten_first_channel(artifact.get("plot_reference_y") if artifact.get("plot_reference_y") is not None else (artifact.get("plot_real_y") if artifact.get("plot_real_y") is not None else (artifact.get("reference_y") if artifact.get("reference_y") is not None else artifact.get("real_y"))))
    recon_y = _flatten_first_channel(artifact.get("plot_reconstructed_y") if artifact.get("plot_reconstructed_y") is not None else artifact.get("reconstructed_y"))

    if real_x is None or recon_x is None:
        raise ValueError(f"Missing x tensors in {artifact_rel}")

    x_axis = list(range(len(real_x)))
    y_axis = list(range(len(real_x), len(real_x) + (len(real_y) if real_y is not None else 0)))

    reference_label = artifact.get("reference_label") or record.get("reference_label") or "reference"
    plt.figure(figsize=(16, 5))
    plt.plot(x_axis, real_x, label=f"{reference_label}_x", linewidth=1.8)
    plt.plot(x_axis, recon_x, label="reconstructed_x", linewidth=1.5)
    if real_y is not None and should_plot_real_y(record, show_idlg_y=show_idlg_y):
        label = f"{reference_label}_y" if str(record.get("name", "")).lower() != "idlg" else f"{reference_label}_y (forced)"
        plt.plot(y_axis, real_y, label=label, linewidth=1.8)
    if recon_y is not None and should_plot_reconstructed_y(record, show_idlg_y=show_idlg_y):
        label = "reconstructed_y" if str(record.get("name", "")).lower() != "idlg" else "reconstructed_y (forced)"
        plt.plot(y_axis, recon_y, label=label, linewidth=1.5)
    plt.axvline(len(real_x) - 1, color="gray", linestyle="--", linewidth=1.0)
    plt.title(
        f"{record.get('name')} client={record.get('client_id')} round={record.get('round_index')} sample={record.get('sample_index')} "
        f"metric={record.get('primary_metric_name', record.get('metric_name'))} value={record.get('primary_metric_value', record.get('mse'))}"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    filename = (
        f"round_{int(record.get('round_index', 0)):04d}_"
        f"client_{record.get('client_id', 'unknown')}_"
        f"sample_{int(record.get('sample_index', 0)):04d}_"
        f"{str(record.get('name', 'attack')).lower()}.png"
    )
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def build_report(records: list[dict[str, Any]], plotted_paths: list[Path]) -> list[str]:
    """Build a small markdown-style report for generated plots."""

    lines = [f"plotted {len(plotted_paths)} attack reconstructions"]
    for record, path in zip(records, plotted_paths):
        lines.append(
            f"- {path.name}: name={record.get('name')} client={record.get('client_id')} round={record.get('round_index')} "
            f"sample={record.get('sample_index')} primary_metric_value={record.get('primary_metric_value', record.get('mse'))} exact_target_mse={record.get('exact_target_mse')}"
        )
    return lines


def main() -> None:
    """Run the CLI that plots saved attack reconstructions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Experiment directory containing attack_results.json and attack_artifacts/")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated PNG files")
    parser.add_argument("--sort-key", default="primary_metric_value", help="Attack record key used for ranking")
    parser.add_argument("--descending", action="store_true", help="Sort descending instead of ascending")
    parser.add_argument("--attack-name", default=None, help="Only plot one attack method, e.g. DLG or iDLG")
    parser.add_argument("--client-id", default=None, help="Only plot one client")
    parser.add_argument("--round-index", type=int, default=None, help="Only plot one round")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of samples to plot")
    parser.add_argument("--show-idlg-y", action="store_true", help="Also plot reconstructed_y for iDLG artifacts")
    args = parser.parse_args()

    run_dir = args.run_dir
    attack_results = load_json(run_dir / "attack_results.json")
    selected = select_records(
        attack_results,
        sort_key=args.sort_key,
        descending=args.descending,
        attack_name=args.attack_name,
        client_id=args.client_id,
        round_index=args.round_index,
        limit=args.limit,
    )
    output_dir = args.output_dir or (run_dir / "attack_plots")
    plotted = [plot_one_artifact(run_dir, record, output_dir, show_idlg_y=args.show_idlg_y) for record in selected]
    report_lines = build_report(selected, plotted)
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
