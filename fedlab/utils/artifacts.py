"""Experiment artifact persistence helpers.

Example:
    Save experiment parameters in the default YAML format::

        from pathlib import Path

        from fedlab.utils.artifacts import save_experiment_config

        paths = save_experiment_config({"training": {"lr": 0.001}}, Path("outputs/run"))
        assert paths[0].name == "config.yaml"
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch

import yaml


SUPPORTED_CONFIG_FORMATS = {"yaml", "yml", "json", "toml"}


def normalize_config_formats(formats: str | Iterable[str] | None) -> list[str]:
    """Normalize requested config artifact formats.

    Args:
        formats: A string such as ``"yaml,json"`` or an iterable like
            ``["yaml", "json"]``. ``None`` defaults to ``["yaml"]``.

    Returns:
        A de-duplicated list of lowercase format names.

    Example:
        >>> normalize_config_formats("yaml,json,yaml")
        ['yaml', 'json']
    """

    if formats is None:
        return ["yaml"]
    if isinstance(formats, str):
        raw_items = formats.split(",")
    else:
        raw_items = list(formats)
    normalized = []
    for item in raw_items:
        fmt = str(item).strip().lower()
        if not fmt:
            continue
        if fmt not in SUPPORTED_CONFIG_FORMATS:
            raise ValueError(f"Unsupported config artifact format: {fmt}")
        if fmt not in normalized:
            normalized.append(fmt)
    return normalized or ["yaml"]


def save_experiment_config(config: dict[str, Any], output_dir: str | Path, formats: str | Iterable[str] | None = None) -> list[Path]:
    """Save experiment parameters in one or more human-readable formats.

    Args:
        config: Experiment configuration mapping.
        output_dir: Directory receiving ``config.<format>`` files.
        formats: Optional format list. Supported values are ``yaml``, ``yml``,
            ``json``, and ``toml``. When omitted, YAML is used.

    Returns:
        Paths written to disk, ordered by requested format.

    Example:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     paths = save_experiment_config({"experiment": {"name": "demo"}}, tmp, ["yaml", "json"])
        ...     [path.name for path in paths]
        ['config.yaml', 'config.json']
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in normalize_config_formats(formats):
        suffix = "yaml" if fmt == "yml" else fmt
        path = output / f"config.{suffix}"
        if suffix == "yaml":
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        elif suffix == "json":
            with path.open("w", encoding="utf-8") as handle:
                json.dump(config, handle, ensure_ascii=False, indent=2)
        elif suffix == "toml":
            path.write_text(_to_toml(config), encoding="utf-8")
        saved.append(path)
    return saved


def _to_toml(data: dict[str, Any]) -> str:
    """Serialize a simple nested mapping to TOML.

    The framework configs use scalar values, lists of scalars, and nested
    sections, which are intentionally the supported surface here.

    Example:
        >>> print(_to_toml({"runtime": {"device": "cpu"}}).strip())
        [runtime]\ndevice = "cpu"
    """

    lines: list[str] = []
    _write_toml_section(lines, [], data)
    return "\n".join(lines).strip() + "\n"


def _write_toml_section(lines: list[str], prefix: list[str], data: dict[str, Any]) -> None:
    """Append one TOML table and recurse into nested tables."""

    scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
    tables = {key: value for key, value in data.items() if isinstance(value, dict)}
    if prefix:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(f"[{'.'.join(prefix)}]")
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    for key, value in tables.items():
        _write_toml_section(lines, [*prefix, str(key)], value)


def _toml_value(value: Any) -> str:
    """Render a Python scalar or scalar list as a TOML value."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    return json.dumps(str(value), ensure_ascii=False)


def _json_ready(value: Any) -> Any:
    """Convert nested artifact payloads into JSON-serializable structures."""

    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def should_save_periodic_artifacts(config: dict[str, Any], round_count: int) -> bool:
    """Return whether periodic artifacts should be saved after the current round count."""

    interval = int(config.get("artifacts", {}).get("save_every_rounds", 0) or 0)
    return interval > 0 and round_count > 0 and round_count % interval == 0


def save_federated_snapshot(
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    snapshot_name: str,
    model_state: Any,
    metrics_history: Any,
    summary: dict[str, Any],
    attack_records: list[dict[str, Any]],
    oracle_model_state: Any | None = None,
    resume_state: dict[str, Any] | None = None,
) -> Path:
    """Persist one round snapshot with the same artifact shape as final outputs."""

    snapshot_dir = Path(output_dir) / "snapshots" / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, snapshot_dir / "model.pt")
    if oracle_model_state is not None:
        torch.save(oracle_model_state, snapshot_dir / "oracle_model.pt")
    if resume_state is not None:
        torch.save(resume_state, snapshot_dir / "resume_state.pt")
    save_experiment_config(config, snapshot_dir, config.get("artifacts", {}).get("config_formats"))
    with (snapshot_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metrics_history), handle, ensure_ascii=False, indent=2)
    with (snapshot_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, ensure_ascii=False, indent=2)
    with (snapshot_dir / "attack_results.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(attack_records), handle, ensure_ascii=False, indent=2)
    return snapshot_dir
