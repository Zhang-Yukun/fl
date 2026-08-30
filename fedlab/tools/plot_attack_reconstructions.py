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

from fedlab.security.registry import compute_recovery_metric_matrix, resolve_recovery_objective
from fedlab.tools.attack_plot_registry import should_plot_real_y, should_plot_reconstructed_y
from fedlab.utils.tracking import _attack_reconstruction_figure


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_records(
    records: list[dict[str, Any]],
    *,
    attack_name: str | None,
    client_id: str | None,
    round_index: int | None,
) -> list[dict[str, Any]]:
    filtered = records
    if attack_name is not None:
        filtered = [record for record in filtered if str(record.get("name")) == attack_name]
    if client_id is not None:
        filtered = [record for record in filtered if str(record.get("client_id")) == client_id]
    if round_index is not None:
        filtered = [record for record in filtered if int(record.get("round_index", -1)) == round_index]
    return filtered


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

    filtered = _filter_records(records, attack_name=attack_name, client_id=client_id, round_index=round_index)

    def key_fn(record: dict[str, Any]) -> tuple[int, float]:
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


def _tensor_batch_size(tensor: Any) -> int | None:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
        return None
    return int(tensor.shape[0])


def _resolve_batch_size(*tensors: Any) -> int:
    for tensor in tensors:
        size = _tensor_batch_size(tensor)
        if size is not None:
            return size
    return 1


def _slice_batch_tensor(tensor: Any, row_index: int, batch_size: int) -> Any:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or batch_size <= 1:
        return tensor
    if int(tensor.shape[0]) != batch_size:
        return tensor
    row = max(0, min(int(row_index), batch_size - 1))
    return tensor[row:row + 1].detach().cpu().clone()


def _compute_row_metrics(
    metric_name: str | None,
    reconstructed_x: Any,
    reference_x: Any,
    *,
    data_range: float = 1.0,
) -> tuple[list[float | None], str | None]:
    if metric_name is None:
        return [], None
    if not isinstance(reconstructed_x, torch.Tensor) or not isinstance(reference_x, torch.Tensor):
        return [], None
    if reconstructed_x.ndim == 0 or reference_x.ndim == 0:
        return [], None
    batch_size = min(int(reconstructed_x.shape[0]), int(reference_x.shape[0]))
    if batch_size <= 0:
        return [], None
    matrix = compute_recovery_metric_matrix(
        reconstructed_x[:batch_size].detach().cpu(),
        reference_x[:batch_size].detach().cpu(),
        str(metric_name),
        float(data_range),
    )
    diagonal = matrix.diagonal()
    return [float(value) for value in diagonal.tolist()], resolve_recovery_objective(None, str(metric_name))


def _build_plot_result(
    record: dict[str, Any],
    artifact: dict[str, Any],
    *,
    row_index: int,
    matched_metric_value: float | None,
    matched_metric_objective: str | None,
    matched_reference_index: int | None,
    show_idlg_y: bool,
) -> SimpleNamespace:
    real_x = _artifact_tensor(artifact, "plot_reference_x", "reference_x")
    recon_x = _artifact_tensor(artifact, "plot_reconstructed_x", "reconstructed_x")
    real_y = _artifact_tensor(artifact, "plot_reference_y", "reference_y")
    recon_y = _artifact_tensor(artifact, "plot_reconstructed_y", "reconstructed_y")
    batch_size = _resolve_batch_size(real_x, recon_x, real_y, recon_y)
    return SimpleNamespace(
        name=record.get('name', artifact.get('name', 'attack')),
        client_id=record.get('client_id', artifact.get('client_id')),
        round_index=record.get('round_index', artifact.get('round_index')),
        sample_index=record.get('sample_index', artifact.get('sample_index')),
        reference_label=artifact.get('reference_label') or record.get('reference_label') or 'reference',
        plot_reference_x=_slice_batch_tensor(real_x, row_index, batch_size),
        plot_reconstructed_x=_slice_batch_tensor(recon_x, row_index, batch_size),
        plot_reference_y=_slice_batch_tensor(real_y, row_index, batch_size) if should_plot_real_y(record, show_policy_overrides=show_idlg_y) else None,
        plot_reconstructed_y=_slice_batch_tensor(recon_y, row_index, batch_size) if should_plot_reconstructed_y(record, show_policy_overrides=show_idlg_y) else None,
        reference_x=_slice_batch_tensor(_artifact_tensor(artifact, "reference_x", "plot_reference_x"), row_index, batch_size),
        reconstructed_x=_slice_batch_tensor(_artifact_tensor(artifact, "reconstructed_x", "plot_reconstructed_x"), row_index, batch_size),
        reference_y=_slice_batch_tensor(_artifact_tensor(artifact, "reference_y", "plot_reference_y"), row_index, batch_size),
        reconstructed_y=_slice_batch_tensor(_artifact_tensor(artifact, "reconstructed_y", "plot_reconstructed_y"), row_index, batch_size),
        matched_reference_metric_name=record.get('matched_reference_metric_name') or artifact.get('matched_reference_metric_name'),
        matched_reference_metric_value=matched_metric_value,
        matched_reference_metric_min_value=matched_metric_value,
        matched_reference_indices=None if matched_reference_index is None else [matched_reference_index],
        matched_metric_objective=matched_metric_objective,
    )


def expand_record_candidates(
    run_dir: Path,
    record: dict[str, Any],
    *,
    show_idlg_y: bool = False,
) -> list[dict[str, Any]]:
    """Expand one artifact record into one candidate per matched sample pair."""

    artifact_rel = record.get("artifact_path")
    if not artifact_rel:
        raise ValueError("Attack record does not include artifact_path")
    artifact = torch.load(run_dir / artifact_rel, map_location="cpu", weights_only=False)

    real_x = _artifact_tensor(artifact, "reference_x", "plot_reference_x")
    recon_x = _artifact_tensor(artifact, "reconstructed_x", "plot_reconstructed_x")
    if real_x is None or recon_x is None:
        raise ValueError(f"Missing x tensors in {artifact_rel}")

    batch_size = _resolve_batch_size(real_x, recon_x)
    metric_name = record.get('matched_reference_metric_name') or artifact.get('matched_reference_metric_name')
    matched_values, matched_objective = _compute_row_metrics(metric_name, recon_x, real_x)
    matched_indices = artifact.get('matched_reference_indices') or record.get('matched_reference_indices') or []
    candidates: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        matched_metric_value = matched_values[row_index] if row_index < len(matched_values) else None
        matched_reference_index = matched_indices[row_index] if row_index < len(matched_indices) else None
        candidates.append({
            'name': record.get('name', artifact.get('name', 'attack')),
            'client_id': record.get('client_id', artifact.get('client_id')),
            'round_index': int(record.get('round_index', artifact.get('round_index', 0)) or 0),
            'sample_index': int(record.get('sample_index', artifact.get('sample_index', 0)) or 0),
            'artifact_path': str(artifact_rel),
            'pair_index': row_index,
            'matched_reference_index': None if matched_reference_index is None else int(matched_reference_index),
            'matched_metric_name': metric_name,
            'matched_metric_value': matched_metric_value,
            'matched_metric_objective': matched_objective,
            'primary_metric_value': record.get('primary_metric_value', record.get('mse')),
            'result': _build_plot_result(
                record,
                artifact,
                row_index=row_index,
                matched_metric_value=matched_metric_value,
                matched_metric_objective=matched_objective,
                matched_reference_index=None if matched_reference_index is None else int(matched_reference_index),
                show_idlg_y=show_idlg_y,
            ),
        })
    return candidates


def _sort_candidates(candidates: list[dict[str, Any]], *, sort_key: str, descending: bool) -> list[dict[str, Any]]:
    if sort_key == 'matched_metric_value':
        groups: dict[str, list[dict[str, Any]]] = {'min': [], 'max': [], 'unknown': []}
        for candidate in candidates:
            objective = candidate.get('matched_metric_objective')
            if objective not in {'min', 'max'}:
                groups['unknown'].append(candidate)
            else:
                groups[str(objective)].append(candidate)
        ordered: list[dict[str, Any]] = []
        ordered.extend(sorted(groups['min'], key=lambda item: (item.get('matched_metric_value') is None, float(item.get('matched_metric_value') or float('inf')))))
        ordered.extend(sorted(groups['max'], key=lambda item: (item.get('matched_metric_value') is None, -float(item.get('matched_metric_value') or float('-inf')))))
        ordered.extend(sorted(groups['unknown'], key=lambda item: (item.get('primary_metric_value') is None, float(item.get('primary_metric_value') or float('inf')))))
        return ordered if not descending else list(reversed(ordered))

    def key_fn(candidate: dict[str, Any]) -> tuple[int, float]:
        value = candidate.get(sort_key)
        if value is None:
            return (1, float('inf'))
        return (0, float(value))

    return sorted(candidates, key=key_fn, reverse=descending)


def select_candidates(
    run_dir: Path,
    records: list[dict[str, Any]],
    *,
    sort_key: str,
    descending: bool,
    attack_name: str | None,
    client_id: str | None,
    round_index: int | None,
    limit: int,
    show_idlg_y: bool = False,
) -> list[dict[str, Any]]:
    filtered = _filter_records(records, attack_name=attack_name, client_id=client_id, round_index=round_index)
    candidates: list[dict[str, Any]] = []
    for record in filtered:
        candidates.extend(expand_record_candidates(run_dir, record, show_idlg_y=show_idlg_y))
    return _sort_candidates(candidates, sort_key=sort_key, descending=descending)[:limit]


def plot_candidate(candidate: dict[str, Any], output_dir: Path) -> Path:
    """Render one expanded candidate into a PNG figure."""

    figure = _attack_reconstruction_figure(candidate['result'])
    if figure is None:
        raise ValueError(f"Could not render attack candidate {candidate['artifact_path']}")
    filename = (
        f"round_{int(candidate.get('round_index', 0)):04d}_"
        f"client_{candidate.get('client_id', 'unknown')}_"
        f"sample_{int(candidate.get('sample_index', 0)):04d}_"
        f"pair_{int(candidate.get('pair_index', 0)):04d}_"
        f"{str(candidate.get('name', 'attack')).lower()}.png"
    )
    output_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_one_artifact(
    run_dir: Path,
    record: dict[str, Any],
    output_dir: Path,
    *,
    show_idlg_y: bool = False,
) -> Path:
    """Render the best candidate from one artifact into a PNG figure."""

    candidates = expand_record_candidates(run_dir, record, show_idlg_y=show_idlg_y)
    selected = _sort_candidates(candidates, sort_key='matched_metric_value', descending=False)
    if not selected:
        raise ValueError(f"No plottable candidates in {record.get('artifact_path')}")
    return plot_candidate(selected[0], output_dir)


def build_report(candidates: list[dict[str, Any]], plotted_paths: list[Path]) -> list[str]:
    """Build a small markdown-style report for generated plots."""

    lines = [f"plotted {len(plotted_paths)} attack reconstructions"]
    for candidate, path in zip(candidates, plotted_paths):
        lines.append(
            f"- {path.name}: name={candidate.get('name')} client={candidate.get('client_id')} round={candidate.get('round_index')} "
            f"sample={candidate.get('sample_index')} pair={candidate.get('pair_index')} matched_reference_index={candidate.get('matched_reference_index')} "
            f"matched_metric_name={candidate.get('matched_metric_name')} matched_metric_value={candidate.get('matched_metric_value')} "
            f"primary_metric_value={candidate.get('primary_metric_value')}"
        )
    return lines


def main() -> None:
    """Run the CLI that plots saved attack reconstructions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Experiment directory containing attack_results.json and attack_artifacts/")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated PNG files")
    parser.add_argument("--sort-key", default="matched_metric_value", help="Expanded candidate key used for ranking")
    parser.add_argument("--descending", action="store_true", help="Sort descending instead of ascending")
    parser.add_argument("--attack-name", default=None, help="Only plot one attack method, e.g. DLG or iDLG")
    parser.add_argument("--client-id", default=None, help="Only plot one client")
    parser.add_argument("--round-index", type=int, default=None, help="Only plot one round")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of matched sample pairs to plot")
    parser.add_argument("--show-idlg-y", action="store_true", help="Also plot reconstructed_y for iDLG artifacts")
    args = parser.parse_args()

    run_dir = args.run_dir
    attack_results = load_json(run_dir / "attack_results.json")
    selected = select_candidates(
        run_dir,
        attack_results,
        sort_key=args.sort_key,
        descending=args.descending,
        attack_name=args.attack_name,
        client_id=args.client_id,
        round_index=args.round_index,
        limit=args.limit,
        show_idlg_y=args.show_idlg_y,
    )
    output_dir = args.output_dir or (run_dir / "attack_plots")
    plotted = [plot_candidate(candidate, output_dir) for candidate in selected]
    report_lines = build_report(selected, plotted)
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
