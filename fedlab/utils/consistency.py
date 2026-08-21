"""Helpers for checking deterministic run equivalence across federated experiment outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_TIMING_KEYS = {
    "time_seconds",
    "avg_time_seconds",
    "total_time_seconds",
    "elapsed_time_seconds",
    "round_time_seconds",
    "epoch_time_seconds",
}


def _should_ignore_key(key: str, ignore_transport: bool) -> bool:
    """Return whether one artifact key should be ignored during comparison."""

    if key in _TIMING_KEYS or key.endswith("time_seconds"):
        return True
    if not ignore_transport:
        return False
    return "transport_" in key or key in {"total_transport_bytes", "total_transport_upload_bytes", "total_transport_download_bytes"}


def _sanitize_artifact(value: Any, ignore_transport: bool = False) -> Any:
    """Remove timing-only fields before deterministic equivalence checks.

    Example:
        ``_sanitize_artifact({"mse": 1.0, "time_seconds": 0.5})`` returns
        ``{"mse": 1.0}``.
    """

    if isinstance(value, dict):
        return {
            key: _sanitize_artifact(item, ignore_transport=ignore_transport)
            for key, item in value.items()
            if not _should_ignore_key(key, ignore_transport)
        }
    if isinstance(value, list):
        return [_sanitize_artifact(item, ignore_transport=ignore_transport) for item in value]
    return value


def _load_json(path: Path) -> Any:
    """Load one JSON artifact from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Load summary, metrics, and attack artifacts for one experiment directory.

    Example:
        ``load_run_artifacts("outputs/check_run")`` returns a mapping with
        ``summary``, ``metrics``, and ``attack_results`` entries.
    """

    path = Path(output_dir)
    return {
        "summary": _load_json(path / "summary.json"),
        "metrics": _load_json(path / "metrics.json"),
        "attack_results": _load_json(path / "attack_results.json") if (path / "attack_results.json").exists() else [],
    }


def _compare_values(left: Any, right: Any, tolerance: float, path: str, diffs: list[str]) -> None:
    """Recursively compare two JSON-like values with numeric tolerance."""

    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left)
        right_keys = set(right)
        for missing in sorted(left_keys - right_keys):
            diffs.append(f"{path}.{missing}: missing on right")
        for missing in sorted(right_keys - left_keys):
            diffs.append(f"{path}.{missing}: missing on left")
        for key in sorted(left_keys & right_keys):
            child = f"{path}.{key}" if path else key
            _compare_values(left[key], right[key], tolerance, child, diffs)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            diffs.append(f"{path}: length {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _compare_values(left_item, right_item, tolerance, f"{path}[{index}]", diffs)
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
            diffs.append(f"{path}: {left} != {right}")
        return
    if left != right:
        diffs.append(f"{path}: {left!r} != {right!r}")


def compare_fedavg_runs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    tolerance: float = 1e-12,
    ignore_transport: bool = False,
) -> list[str]:
    """Return deterministic-equivalence mismatches between two experiment directories.

    Example:
        ``compare_fedavg_runs("outputs/sync", "outputs/async")`` returns an
        empty list when all non-timing artifacts agree.
    """

    left = _sanitize_artifact(load_run_artifacts(reference_dir), ignore_transport=ignore_transport)
    right = _sanitize_artifact(load_run_artifacts(candidate_dir), ignore_transport=ignore_transport)
    diffs: list[str] = []
    _compare_values(left, right, tolerance=tolerance, path="", diffs=diffs)
    return diffs
