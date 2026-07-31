"""Configuration loading with nested YAML includes and CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml
from loguru import logger


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_one(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _resolve_includes(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve YAML files listed under the optional ``includes`` field."""

    includes = data.pop("includes", [])
    if isinstance(includes, (str, Path)):
        includes = [includes]
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = base_dir / include_path
        merged = _deep_merge(merged, _resolve_includes(_load_one(include_path), include_path.parent))
    return _deep_merge(merged, data)


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply CLI overrides such as ``training.lr=0.001`` to a config copy."""

    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: {item}")
        key, value = item.split("=", 1)
        _set_dotted(result, key, _parse_scalar(value))
    return result


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolve nested includes, and apply CLI overrides."""

    path = Path(path).expanduser().resolve()
    config = _resolve_includes(_load_one(path), path.parent)
    config = apply_overrides(config, overrides)
    logger.info("Loaded config from {}", path)
    return config

