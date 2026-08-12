"""Task registry for task-agnostic model, data, and metric resolution."""

from __future__ import annotations

from typing import Any

from torch import nn

from federated_ts.tasks.base import TaskSpec
from federated_ts.tasks.forecasting import FORECASTING_TASK


_TASKS: dict[str, TaskSpec] = {
    "forecasting": FORECASTING_TASK,
    "time_series_forecasting": FORECASTING_TASK,
    "rare_earth_forecasting": FORECASTING_TASK,
}


def task_type(config: dict[str, Any]) -> str:
    """Resolve the configured task type, defaulting to forecasting."""

    task_cfg = config.get("task", {})
    value = task_cfg.get("type")
    if value is None:
        return "forecasting"
    return str(value).lower()


def get_task(config: dict[str, Any]) -> TaskSpec:
    """Return the registered task plugin for one experiment config."""

    name = task_type(config)
    if name not in _TASKS:
        raise ValueError(f"Unknown task type: {name}")
    return _TASKS[name]


def primary_metric(config: dict[str, Any]) -> str:
    """Return the metric name used for best-checkpoint and early-stop logic."""

    return get_task(config).primary_metric


def primary_metric_mode(config: dict[str, Any]) -> str:
    """Return whether smaller or larger values are considered better."""

    return get_task(config).primary_metric_mode


def annotate_model(model: nn.Module, config: dict[str, Any]) -> nn.Module:
    """Attach resolved task metadata to a built model for generic loops."""

    task = get_task(config)
    setattr(model, "_federated_task_type", task.name)
    setattr(model, "_federated_task_config", config)
    return model


def get_model_task(model: nn.Module) -> TaskSpec:
    """Resolve the task plugin attached to a model, falling back to forecasting."""

    name = getattr(model, "_federated_task_type", "forecasting")
    if name not in _TASKS:
        raise ValueError(f"Unknown model task type: {name}")
    return _TASKS[name]


def get_model_config(model: nn.Module) -> dict[str, Any]:
    """Return the config attached to a model if available."""

    return getattr(model, "_federated_task_config", {})
