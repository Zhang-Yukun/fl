#!/usr/bin/env python3
"""Batch-plot attack reconstructions from multiple experiment directories.

Example:
    python -m fedlab.tools.plot_attack_batch_suite \
        outputs/repro_pat50_attackviz_50_reconmse/fedavg_seed2026_pat50_payloadv1 \
        outputs/repro_pat50_attackviz_50_reconmse/secure_qint8_bidir_seed2026_pat50_payloadv1 \
        outputs/repro_pat50_attackviz_50_reconmse/topk_fedavg_seed2026_pat50_run1 \
        --output-dir outputs/repro_pat50_attackviz_50_reconmse/attack_gallery
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fedlab.tools.attack_plot_registry import discover_attack_names
from fedlab.tools.plot_attack_reconstructions import plot_one_artifact, select_records


def load_json(path: Path) -> Any:
    """Load one JSON file from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_algorithm_label(run_dir: Path) -> str:
    """Resolve a readable algorithm label from one experiment directory."""

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        if isinstance(metrics, list) and metrics:
            value = metrics[-1].get("algorithm")
            if value:
                return str(value)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        value = summary.get("algorithm")
        if value:
            return str(value)
    return run_dir.name


def sanitize_label(value: str) -> str:
    """Return a filesystem-safe folder label."""

    keep: list[str] = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def plot_attack_suite(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    limit_per_attack: int = 10,
    sort_key: str = "mse",
    descending: bool = False,
    attack_names: tuple[str, ...] | None = None,
) -> list[str]:
    """Plot a fixed number of reconstructions per attack for each experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [f"plotted attack galleries for {len(run_dirs)} run(s)"]
    for run_dir in run_dirs:
        attack_results_path = run_dir / "attack_results.json"
        if not attack_results_path.exists():
            report_lines.append(f"- skipped {run_dir}: missing attack_results.json")
            continue
        records = load_json(attack_results_path)
        algorithm_label = sanitize_label(load_algorithm_label(run_dir))
        selected_attack_names = attack_names or discover_attack_names(records)
        report_lines.append(f"- {run_dir.name}: algorithm={algorithm_label}")
        for attack_name in selected_attack_names:
            selected = select_records(
                records,
                sort_key=sort_key,
                descending=descending,
                attack_name=attack_name,
                client_id=None,
                round_index=None,
                limit=limit_per_attack,
            )
            attack_dir = output_dir / algorithm_label / attack_name
            attack_dir.mkdir(parents=True, exist_ok=True)
            plotted_names: list[str] = []
            for record in selected:
                generated = plot_one_artifact(run_dir, record, attack_dir)
                target = attack_dir / f"{run_dir.name}__{generated.name}"
                if generated != target:
                    generated.rename(target)
                plotted_names.append(target.name)
            (attack_dir / f"{run_dir.name}_report.md").write_text(
                "\n".join(
                    [
                        f"run={run_dir.name}",
                        f"algorithm={algorithm_label}",
                        f"attack={attack_name}",
                        f"count={len(plotted_names)}",
                        *[f"- {name}" for name in plotted_names],
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report_lines.append(f"  - {attack_name}: {len(plotted_names)} plot(s)")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_lines


def main() -> None:
    """Run the batch plotting CLI for multiple experiment folders."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Experiment directories containing attack artifacts")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root output directory for grouped plots")
    parser.add_argument("--limit-per-attack", type=int, default=11, help="Number of plots per attack type and run")
    parser.add_argument("--sort-key", default="mse", help="Attack record key used for ranking")
    parser.add_argument("--descending", action="store_true", help="Sort descending instead of ascending")
    args = parser.parse_args()

    lines = plot_attack_suite(
        args.run_dirs,
        args.output_dir,
        limit_per_attack=args.limit_per_attack,
        sort_key=args.sort_key,
        descending=args.descending,
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
