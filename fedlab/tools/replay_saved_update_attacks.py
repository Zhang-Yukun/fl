#!/usr/bin/env python3
"""Replay DLG/iDLG attacks from saved server-visible update captures.

Example:
    python -m fedlab.tools.replay_saved_update_attacks outputs/demo_run
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from loguru import logger

from fedlab.federated.algorithms import (
    _execute_attack_round_task,
    _resolve_attack_device,
    build_update_attack_round_task,
    load_captured_update_records,
)
from fedlab.security.attacks import save_attack_artifacts, summarize_attack_results
from fedlab.utils.artifacts import save_experiment_config
from fedlab.utils.config import load_config
from fedlab.utils.logging import setup_logging


def _default_config_path(run_dir: Path) -> Path:
    """Return the config artifact path stored alongside one experiment run."""

    for name in ("config.yaml", "config.yml", "config.json", "config.toml"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No saved config artifact found under {run_dir}")


def _round_indices(records: list[dict[str, object]]) -> list[int]:
    """Return sorted unique round indices present in saved update captures."""

    return sorted({int(record["round_index"]) for record in records})


def main() -> None:
    """Run the CLI that replays update-payload attacks from saved captures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Experiment directory containing config.yaml and saved_updates/")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory receiving replayed attack artifacts")
    parser.add_argument("--config", type=Path, default=None, help="Optional config path overriding run_dir/config.yaml")
    parser.add_argument("--override", action="append", default=[], help="Optional config overrides applied before replay")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config_path = (args.config.expanduser().resolve() if args.config is not None else _default_config_path(run_dir))
    config = load_config(config_path, args.override)
    replay_config = copy.deepcopy(config)
    replay_config.setdefault("attack", {})["enabled"] = True
    target_type = str(replay_config.get("attack", {}).get("target_type", "update_payload")).lower()
    if target_type != "update_payload":
        raise ValueError(
            "Saved-update replay currently supports attack.target_type=update_payload only; "
            f"received {target_type!r}"
        )

    records = load_captured_update_records(run_dir)
    if not records:
        raise FileNotFoundError(f"No saved update captures found under {run_dir / 'saved_updates'}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else (run_dir / "offline_attack_replay")
    setup_logging(output_dir, replay_config.get("runtime", {}).get("log_level", "INFO"))
    save_experiment_config(replay_config, output_dir, replay_config.get("artifacts", {}).get("config_formats"))

    attack_device = _resolve_attack_device(replay_config)
    max_rounds = int(replay_config.get("federated", {}).get("rounds", 0) or (max(_round_indices(records)) + 1))
    attack_results = []
    for round_index in _round_indices(records):
        task = build_update_attack_round_task(replay_config, records, round_index, max_rounds)
        if task is None:
            continue
        round_result = _execute_attack_round_task(replay_config, task, attack_device)
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
        float(replay_config.get("attack", {}).get("success_rate_threshold", 0.03)),
    )
    with (output_dir / "attack_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(attack_summary, handle, ensure_ascii=False, indent=2)

    payload = {
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "attack_count": len(attack_records),
        "attack_summary_path": str(output_dir / "attack_summary.json"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
