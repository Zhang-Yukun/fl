#!/usr/bin/env python3
"""Plot saved attack reconstructions against the corresponding real series.

Example:
    python -m fedlab.tools.plot_attack_reconstructions         outputs/repro_pat50_attackviz_200/fedavg_seed2026_pat50_payloadv1         --output-dir outputs/repro_pat50_attackviz_200/fedavg_seed2026_pat50_payloadv1/attack_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fedlab.tools.attack_plot_registry import should_plot_real_y, should_plot_reconstructed_y
from fedlab.utils.tracking import _attack_reconstruction_figure


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




def _artifact_tensor(payload: dict[str, Any], *names: str):
    """Return the first available tensor payload from a saved artifact."""

    for name in names:
        value = payload.get(name)
        if value is not None:
            return value
    return None


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

    real_x = _artifact_tensor(artifact, "plot_reference_x", "plot_real_x", "reference_x", "real_x")
    recon_x = _artifact_tensor(artifact, "plot_reconstructed_x", "reconstructed_x")
    real_y = _artifact_tensor(artifact, "plot_reference_y", "plot_real_y", "reference_y", "real_y")
    recon_y = _artifact_tensor(artifact, "plot_reconstructed_y", "reconstructed_y")

    if real_x is None or recon_x is None:
        raise ValueError(f"Missing x tensors in {artifact_rel}")

    result = SimpleNamespace(
        name=record.get('name', artifact.get('name', 'attack')),
        client_id=record.get('client_id', artifact.get('client_id')),
        round_index=record.get('round_index', artifact.get('round_index')),
        sample_index=record.get('sample_index', artifact.get('sample_index')),
        reference_label=artifact.get('reference_label') or record.get('reference_label') or 'reference',
        plot_reference_x=real_x,
        plot_reconstructed_x=recon_x,
        plot_reference_y=real_y if should_plot_real_y(record, show_policy_overrides=show_idlg_y) else None,
        plot_reconstructed_y=recon_y if should_plot_reconstructed_y(record, show_policy_overrides=show_idlg_y) else None,
        reference_x=real_x,
        reconstructed_x=recon_x,
        reference_y=real_y,
        reconstructed_y=recon_y,
    )
    figure = _attack_reconstruction_figure(result)
    if figure is None:
        raise ValueError(f"Could not render attack artifact {artifact_rel}")

    filename = (
        f"round_{int(record.get('round_index', 0)):04d}_"
        f"client_{record.get('client_id', 'unknown')}_"
        f"sample_{int(record.get('sample_index', 0)):04d}_"
        f"{str(record.get('name', 'attack')).lower()}.png"
    )
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def build_report(records: list[dict[str, Any]], plotted_paths: list[Path]) -> list[str]:
    """Build a small markdown-style report for generated plots."""

    lines = [f"plotted {len(plotted_paths)} attack reconstructions"]
    for record, path in zip(records, plotted_paths):
        lines.append(
            f"- {path.name}: name={record.get('name')} client={record.get('client_id')} round={record.get('round_index')} "
            f"sample={record.get('sample_index')} primary_metric_value={record.get('primary_metric_value', record.get('mse'))} nearest_client_train_mse={record.get('nearest_client_train_mse')}"
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
