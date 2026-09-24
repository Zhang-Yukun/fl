"""Evaluate reconstructed attack tensors against client training data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

import torch

from fedlab.attack.data import ClientReferenceSet, load_client_reference_set
from fedlab.security.attack_common import (
    AttackResult,
    apply_set_recovery_metrics,
    save_attack_artifacts,
    summarize_attack_results,
)
from fedlab.utils.artifacts import save_experiment_config
from fedlab.utils.config import load_config
from fedlab.utils.logging import setup_logging


ReferenceLoader = Callable[[dict[str, Any], str], ClientReferenceSet]


def default_evaluation_config_path(attack_dir: Path) -> Path:
    """Return the saved configuration belonging to an attack execution."""

    for name in ("config.yaml", "config.yml", "config.json", "config.toml"):
        candidate = attack_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No saved attack config found under {attack_dir}")


def _inverse_plot_tensor(
    values: torch.Tensor | None,
    mean: list[float] | None,
    std: list[float] | None,
) -> torch.Tensor | None:
    """Restore standardized forecasting tensors for visualization."""

    if values is None:
        return None
    if mean is None or std is None:
        return values.detach().cpu().clone()
    tensor = values.detach().cpu().to(torch.float32)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).reshape(1, 1, -1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).reshape(1, 1, -1)
    while mean_tensor.ndim < tensor.ndim:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    return tensor * std_tensor + mean_tensor


def _attack_result_from_payload(payload: dict[str, Any], artifact_path: str) -> AttackResult:
    """Rebuild an attack result from one unevaluated tensor artifact."""

    objective_mse = payload.get("objective_mse", payload.get("primary_metric_value"))
    if objective_mse is None:
        raise ValueError(f"Attack artifact {artifact_path} does not contain objective_mse")
    return AttackResult(
        name=str(payload["name"]),
        mse=float(objective_mse),
        iterations=int(payload.get("iterations", 0)),
        time_seconds=float(payload.get("time_seconds", 0.0)),
        success=False,
        success_threshold=float("nan"),
        gradient_mse=float(objective_mse),
        target_type=str(payload.get("target_type", "update_payload")),
        metric_name="objective_mse",
        client_id=None if payload.get("client_id") is None else str(payload["client_id"]),
        round_index=None if payload.get("round_index") is None else int(payload["round_index"]),
        sample_index=None if payload.get("sample_index") is None else int(payload["sample_index"]),
        artifact_path=artifact_path,
        reconstructed_x=payload.get("reconstructed_x"),
        reconstructed_y=payload.get("reconstructed_y"),
    )


def load_unevaluated_attack_results(attack_dir: Path) -> list[AttackResult]:
    """Load raw reconstruction artifacts produced by the attack-only phase."""

    records_path = attack_dir / "attack_results.json"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing attack result index: {records_path}")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    results = []
    for record in records:
        relative_path = record.get("artifact_path")
        if not relative_path:
            raise ValueError("Attack result record is missing artifact_path")
        payload = torch.load(attack_dir / relative_path, map_location="cpu", weights_only=False)
        results.append(_attack_result_from_payload(payload, str(relative_path)))
    return results


def evaluate_attack_results(
    config: dict[str, Any],
    results: list[AttackResult],
    *,
    reference_loader: ReferenceLoader = load_client_reference_set,
) -> list[AttackResult]:
    """Attach reference matching, recovery metrics, and plot tensors."""

    reference_cache: dict[str, ClientReferenceSet] = {}

    def get_reference(client_id: str) -> ClientReferenceSet:
        if client_id not in reference_cache:
            reference_cache[client_id] = reference_loader(config, client_id)
        return reference_cache[client_id]

    grouped: dict[tuple[str, int | None, str], list[AttackResult]] = {}
    for result in results:
        if result.client_id is None:
            raise ValueError("Cannot evaluate an attack artifact without client_id")
        grouped.setdefault((result.client_id, result.round_index, result.name), []).append(result)

    for (client_id, _round_index, _method), subset in grouped.items():
        reference = get_reference(client_id)
        apply_set_recovery_metrics(
            subset,
            reference_inputs=reference.inputs,
            reference_targets=reference.targets,
            config=config,
        )
        for result in subset:
            result.plot_reference_x = _inverse_plot_tensor(
                result.reference_x, reference.scale_mean, reference.scale_std
            )
            result.plot_reconstructed_x = _inverse_plot_tensor(
                result.reconstructed_x, reference.scale_mean, reference.scale_std
            )
            result.plot_reference_y = _inverse_plot_tensor(
                result.reference_y, reference.scale_mean, reference.scale_std
            )
            result.plot_reconstructed_y = _inverse_plot_tensor(
                result.reconstructed_y, reference.scale_mean, reference.scale_std
            )
    return results


def evaluate_saved_attacks(
    attack_dir: Path,
    *,
    output_dir: Path,
    config_path: Path,
    overrides: list[str],
) -> dict[str, object]:
    """Evaluate an attack-only output directory using training references."""

    config = load_config(config_path, overrides)
    setup_logging(output_dir, config.get("runtime", {}).get("log_level", "INFO"))
    save_experiment_config(config, output_dir, config.get("artifacts", {}).get("config_formats"))
    results = evaluate_attack_results(config, load_unevaluated_attack_results(attack_dir))
    records = save_attack_artifacts(output_dir, results)
    with (output_dir / "attack_results.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    attack_cfg = config.get("attack", {})
    summary = summarize_attack_results(
        results,
        float(attack_cfg.get("success_rate_threshold", 0.05)),
        float(
            attack_cfg.get(
                "overall_success_rate_threshold",
                attack_cfg.get("success_rate_threshold", 0.05),
            )
        ),
    )
    summary["evaluation_status"] = "complete"
    with (output_dir / "attack_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    execution_path = attack_dir / "attack_execution_summary.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.exists() else {}
    source_run_dir = execution.get("source_run_dir")
    if source_run_dir:
        from fedlab.attack.replay import build_replay_summary

        run_summary = build_replay_summary(Path(str(source_run_dir)), records, summary)
    else:
        run_summary = {
            "attack_evaluations": len(records),
            "attack_summary": copy.deepcopy(summary),
        }
    run_summary["source_attack_dir"] = str(attack_dir)
    run_summary["source_run_dir"] = source_run_dir
    run_summary["evaluation_status"] = "complete"
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    return {
        "source_attack_dir": str(attack_dir),
        "output_dir": str(output_dir),
        "attack_count": len(records),
        "evaluation_status": "complete",
        "attack_summary_path": str(output_dir / "attack_summary.json"),
        "summary_path": str(output_dir / "summary.json"),
    }


def run_evaluation_cli() -> None:
    """Parse CLI arguments and evaluate saved reconstruction artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attack_dir", type=Path, help="Attack-only output containing attack_artifacts/")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory receiving evaluated artifacts")
    parser.add_argument("--config", type=Path, default=None, help="Optional evaluation configuration")
    parser.add_argument("--override", action="append", default=[], help="Configuration override")
    args = parser.parse_args()
    attack_dir = args.attack_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else default_evaluation_config_path(attack_dir)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else attack_dir / "evaluation"
    payload = evaluate_saved_attacks(
        attack_dir,
        output_dir=output_dir,
        config_path=config_path,
        overrides=list(args.override),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
