"""Offline replay entry points for saved-update attack execution."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Sequence

from loguru import logger

from fedlab.replay_capture.artifacts import load_captured_update_records
from fedlab.attack.runner import execute_attack_round_task, resolve_attack_device
from fedlab.attack.tasks import build_update_attack_round_task
from fedlab.utils.runtime import configure_random_seed, configure_torch_runtime
from fedlab.security.attack_common import save_attack_artifacts, summarize_attack_results
from fedlab.utils.artifacts import save_experiment_config
from fedlab.utils.config import load_config
from fedlab.utils.logging import setup_logging


def default_config_path(run_dir: Path) -> Path:
    """Return the config artifact path stored alongside one experiment run."""

    for name in ("config.yaml", "config.yml", "config.json", "config.toml"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No saved config artifact found under {run_dir}")


def round_indices(records: list[dict[str, object]]) -> list[int]:
    """Return sorted unique round indices present in saved update captures."""

    return sorted({int(record["round_index"]) for record in records})


def build_replay_config(config_path: Path, overrides: list[str], forced_methods: Sequence[str] | None) -> dict:
    """Load one replay config and optionally force the executed attack methods."""

    config = load_config(config_path, overrides)
    replay_config = copy.deepcopy(config)
    replay_config.setdefault("attack", {})["enabled"] = True
    if forced_methods is not None:
        replay_config["attack"]["methods"] = list(forced_methods)
    target_type = str(replay_config.get("attack", {}).get("target_type", "update_payload")).lower()
    if target_type != "update_payload":
        raise ValueError(
            "Saved-update replay currently supports attack.target_type=update_payload only; "
            f"received {target_type!r}"
        )
    return replay_config


def build_replay_summary(
    run_dir: Path,
    attack_records: list[dict[str, object]],
    attack_summary: dict[str, object],
) -> dict[str, object]:
    """Copy the source run summary and refresh the attack-related fields for replay."""

    summary_path = run_dir / "summary.json"
    summary: dict[str, object] = {}
    if summary_path.exists():
        summary = copy.deepcopy(json.loads(summary_path.read_text(encoding="utf-8")))
    summary["attack_target_type"] = attack_summary.get(
        "target_type",
        summary.get("attack_target_type", "update_payload"),
    )
    summary["attack_primary_metric_name"] = attack_summary["primary_metric_name"]
    summary["attack_primary_metric_direction"] = attack_summary["primary_metric_direction"]
    summary["attack_overall_avg_primary_metric_value"] = attack_summary["overall_avg_primary_metric_value"]
    summary["attack_overall_best_primary_metric_value"] = attack_summary["overall_best_primary_metric_value"]
    summary["attack_success_rate"] = attack_summary["overall_success_rate"]
    summary["attack_evaluations"] = len(attack_records)
    summary["attack_summary"] = copy.deepcopy(attack_summary)
    return summary


def replay_saved_update_attacks(
    run_dir: Path,
    *,
    output_dir: Path,
    config_path: Path,
    overrides: list[str],
    forced_methods: Sequence[str] | None = None,
) -> dict[str, object]:
    """Replay saved-update attacks and persist artifacts under one output directory."""

    replay_config = build_replay_config(config_path, overrides, forced_methods)
    records = load_captured_update_records(run_dir)
    if not records:
        raise FileNotFoundError(f"No saved update captures found under {run_dir / 'saved_updates'}")
    setup_logging(output_dir, replay_config.get("runtime", {}).get("log_level", "INFO"))
    save_experiment_config(replay_config, output_dir, replay_config.get("artifacts", {}).get("config_formats"))
    configure_torch_runtime(replay_config)
    configure_random_seed(replay_config)
    if replay_config.get("attack", {}).get("seed") is None:
        logger.warning(
            "attack.seed is unset in replay config; replayed attack records will match online selection order, "
            "but optimization metrics may drift if the original online run consumed random numbers before attacking"
        )
    attack_device = resolve_attack_device(replay_config)
    max_rounds = int(replay_config.get("federated", {}).get("rounds", 0) or (max(round_indices(records)) + 1))
    attack_results = []
    for round_index in round_indices(records):
        task = build_update_attack_round_task(replay_config, records, round_index, max_rounds)
        if task is None:
            continue
        round_result = execute_attack_round_task(replay_config, task, attack_device)
        attack_results.extend(round_result.attacks)
        logger.info(
            "Replayed round {} with {} attacks on {}",
            round_index,
            len(round_result.attacks),
            attack_device,
        )
    attack_records = save_attack_artifacts(output_dir, attack_results)
    with (output_dir / "attack_results.json").open("w", encoding="utf-8") as handle:
        json.dump(attack_records, handle, ensure_ascii=False, indent=2)
    attack_summary = summarize_attack_results(
        attack_results,
        float(replay_config.get("attack", {}).get("success_rate_threshold", 0.05)),
        float(
            replay_config.get("attack", {}).get(
                "overall_success_rate_threshold",
                replay_config.get("attack", {}).get("success_rate_threshold", 0.05),
            )
        ),
    )
    with (output_dir / "attack_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(attack_summary, handle, ensure_ascii=False, indent=2)
    replay_summary = build_replay_summary(run_dir, attack_records, attack_summary)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(replay_summary, handle, ensure_ascii=False, indent=2)
    return {
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "attack_count": len(attack_records),
        "attack_summary_path": str(output_dir / "attack_summary.json"),
        "summary_path": str(output_dir / "summary.json"),
    }


def run_replay_cli(*, description: str, forced_methods: Sequence[str] | None = None) -> None:
    """Run the replay CLI with an optional forced attack-method subset."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("run_dir", type=Path, help="Experiment directory containing config.yaml and saved_updates/")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory receiving replayed attack artifacts")
    parser.add_argument("--config", type=Path, default=None, help="Optional config path overriding run_dir/config.yaml")
    parser.add_argument("--override", action="append", default=[], help="Optional config overrides applied before replay")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config is not None else default_config_path(run_dir)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else (run_dir / "offline_attack_replay")
    payload = replay_saved_update_attacks(
        run_dir,
        output_dir=output_dir,
        config_path=config_path,
        overrides=list(args.override),
        forced_methods=forced_methods,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
