"""Configuration loading with nested YAML includes and CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable


_RUNTIME_DEFAULTS = {
    "experiment": {
        "mode": "federated",
    },
    "runtime": {
        "device": "cpu",
        "log_level": "INFO",
        "deterministic": True,
    },
    "task": {
        "type": "forecasting",
    },
    "training": {
        "lr": 1e-3,
        "patience": 50,
        "min_delta": 0.0,
        "loss": "mse",
        "optimizer": "adam",
        "optimizer_eps": 1e-8,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "nesterov": False,
    },
    "evaluation": {
        "mode": "protocol",
        "metrics": ["mse", "mae", "mape"],
    },
    "centralized": {
        "rounds": 10,
    },
    "federated": {
        "algorithm": "fedavg",
        "rounds": 20,
        "local_epochs": 1,
    },
    "transport": {
        "upload_mode": "update",
        "download_mode": "model",
    },
    "attack": {
        "enabled": True,
        "target_type": "update_payload",
        "reference_metric": "nearest_client_train_mse",
        "report_metrics": ["nearest_client_train_mse"],
        "steps": 300,
        "lr": 0.001,
        "optimizer": "adam",
        "restarts": 1,
        "lbfgs_history_size": 20,
        "success_mse_threshold": 0.5,
        "success_rate_threshold": 0.03,
        "data_range": 1.0,
        "client_selection": "all",
        "clients_per_round": 1,
        "frequency_rounds": 1,
        "sample_count": 1,
        "max_samples": 1,
        "model_mode": "train",
        "local_optimizer": "adam",
        "local_lr": 1e-3,
        "local_optimizer_eps": 1e-8,
        "async_enabled": False,
        "async_workers": 1,
        "async_max_pending_rounds": 5,
        "device": "same",
    },
    "tracking": {
        "enabled": True,
        "offline": True,
        "project": "federated-rare-earth",
    },
    "grpc": {
        "address": "0.0.0.0:50051",
        "server_address": "127.0.0.1:50051",
        "poll_seconds": 1.0,
        "max_message_mb": 256.0,
    },
    "artifacts": {
        "config_formats": ["yaml"],
        "save_every_rounds": 0,
    },
}


_COMMON_FEDERATED_KEYS = {"algorithm", "rounds", "local_epochs", "local_steps"}
_ALGORITHM_FEDERATED_KEYS = {
    "compressed_fedavg": {"topk_fraction"},
    "sparse_fedavg": {"topk_fraction"},
    "dp_topk_fedavg": {"topk_fraction"},
    "randomk_fedavg": {"topk_fraction", "randomk_seed"},
    "soteriafl": {"topk_fraction", "randomk_seed"},
    "secure_quantized_fedavg": {"quantization_dtype", "quantization_stochastic_rounding", "quantization_seed"},
    "qsgd_fedavg": {"qsgd_levels", "quantization_seed"},
    "sign_fedavg": set(),
    "fedavg": set(),
    "fedaware": set(),
    "adaptive_clipped_rdp_fedavg": set(),
    "ega_fedavg": {"quantization_seed"},
}
_ALGORITHM_ROOT_BLOCKS = {
    "fedaware": {"fedaware"},
    "adaptive_clipped_rdp_fedavg": {"adaptive_clipped_rdp"},
    "ega_fedavg": {"ega"},
}
_ALGORITHM_PRIVACY_USERS = {"dp_topk_fedavg", "soteriafl", "secure_quantized_fedavg"}

import yaml
from loguru import logger


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings without mutating either input."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_one(path: Path) -> dict[str, Any]:
    """Load one YAML file and validate that its root is a mapping."""

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
    """Parse a command-line override value using YAML scalar rules."""

    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested mapping value addressed by a dotted key."""

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


def _materialize_runtime_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Merge runtime defaults so saved configs match actual effective behavior."""

    return _deep_merge(_RUNTIME_DEFAULTS, config)


def _validate_no_deprecated_schedule_keys(config: dict[str, Any]) -> None:
    """Reject deprecated schedule keys that were intentionally removed."""

    centralized_cfg = config.get("centralized", {})
    if isinstance(centralized_cfg, dict) and centralized_cfg.get("epochs") is not None:
        raise ValueError("Deprecated config key centralized.epochs is no longer supported; use centralized.rounds")
    training_cfg = config.get("training", {})
    if isinstance(training_cfg, dict) and training_cfg.get("epochs") is not None:
        raise ValueError("Deprecated config key training.epochs is no longer supported")


def _sanitize_algorithm_config(config: dict[str, Any]) -> dict[str, Any]:
    """Drop algorithm-specific config blocks that do not apply to the active method."""

    result = copy.deepcopy(config)
    federated_cfg = result.get("federated")
    if not isinstance(federated_cfg, dict):
        return result
    algorithm = str(federated_cfg.get("algorithm", "")).lower()
    if not algorithm or algorithm not in _ALGORITHM_FEDERATED_KEYS:
        return result

    allowed_federated_keys = _COMMON_FEDERATED_KEYS | _ALGORITHM_FEDERATED_KEYS.get(algorithm, set())
    result["federated"] = {
        key: copy.deepcopy(value)
        for key, value in federated_cfg.items()
        if key in allowed_federated_keys
    }

    active_blocks = _ALGORITHM_ROOT_BLOCKS.get(algorithm, set())
    for block_name in _ALGORITHM_ROOT_BLOCKS.values():
        for key in block_name:
            if key not in active_blocks:
                result.pop(key, None)

    if algorithm not in _ALGORITHM_PRIVACY_USERS:
        result.pop("privacy", None)

    return result


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolve nested includes, and apply CLI overrides."""

    path = Path(path).expanduser().resolve()
    config = _resolve_includes(_load_one(path), path.parent)
    config = apply_overrides(config, overrides)
    config = _materialize_runtime_defaults(config)
    _validate_no_deprecated_schedule_keys(config)
    config = _sanitize_algorithm_config(config)
    logger.info("Loaded config from {}", path)
    return config

