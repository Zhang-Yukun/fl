"""Task registry for task-agnostic model, data, loss, metric, and optimizer resolution."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

from fedlab.tasks.base import LossFn, MetricFn, TaskSpec
from fedlab.utils.metrics import accuracy, cross_entropy, mae, mape, mse


_TASKS: dict[str, TaskSpec] = {}
_BUILTIN_TASKS_LOADED = False


def _normalize_name(name: str) -> str:
    """Normalize registry keys to the canonical lowercase form."""

    return str(name).strip().lower()


def register_task(task: TaskSpec, *aliases: str, replace: bool = False) -> TaskSpec:
    """Register one task plugin and optional aliases."""

    names = (_normalize_name(task.name), *(_normalize_name(alias) for alias in aliases))
    for name in names:
        existing = _TASKS.get(name)
        if existing is not None and existing is not task and not replace:
            raise ValueError(f"Task alias already registered: {name}")
    for name in names:
        _TASKS[name] = task
    return task


def task_plugin(*aliases: str, replace: bool = False):
    """Decorator for registering one ``TaskSpec`` as a task plugin."""

    def decorator(task: TaskSpec) -> TaskSpec:
        return register_task(task, *aliases, replace=replace)

    return decorator


def list_registered_tasks() -> dict[str, TaskSpec]:
    """Return a snapshot of currently registered tasks."""

    _ensure_builtin_tasks_registered()
    return dict(_TASKS)


def _ensure_builtin_tasks_registered() -> None:
    """Load builtin task modules once so they can self-register."""

    global _BUILTIN_TASKS_LOADED
    if _BUILTIN_TASKS_LOADED:
        return
    _BUILTIN_TASKS_LOADED = True
    from fedlab.tasks import classification as _classification  # noqa: F401
    from fedlab.tasks import forecasting as _forecasting  # noqa: F401


def _loss_mse(_config: dict[str, Any]) -> LossFn:
    """Build the mean-squared-error training loss."""

    return nn.MSELoss()


def _loss_l1(_config: dict[str, Any]) -> LossFn:
    """Build the mean-absolute-error training loss."""

    return nn.L1Loss()


def _loss_smooth_l1(config: dict[str, Any]) -> LossFn:
    """Build SmoothL1 loss using the configured beta value."""

    beta = float(config.get("training", {}).get("smooth_l1_beta", 1.0))
    return nn.SmoothL1Loss(beta=beta)


def _loss_huber(config: dict[str, Any]) -> LossFn:
    """Build Huber loss using the configured delta value."""

    delta = float(config.get("training", {}).get("huber_delta", 1.0))
    return nn.HuberLoss(delta=delta)


def _loss_cross_entropy(_config: dict[str, Any]) -> LossFn:
    """Build categorical cross-entropy for integer classification labels."""

    return nn.CrossEntropyLoss()


LOSS_REGISTRY: dict[str, callable] = {
    "mse": _loss_mse,
    "l2": _loss_mse,
    "mae": _loss_l1,
    "l1": _loss_l1,
    "smooth_l1": _loss_smooth_l1,
    "huber": _loss_huber,
    "cross_entropy": _loss_cross_entropy,
    "ce": _loss_cross_entropy,
}

METRIC_REGISTRY: dict[str, MetricFn] = {
    "mse": lambda pred, target: {"mse": mse(pred, target)},
    "mae": lambda pred, target: {"mae": mae(pred, target)},
    "mape": lambda pred, target: {"mape": mape(pred, target)},
    "accuracy": lambda pred, target: {"accuracy": accuracy(pred, target)},
    "cross_entropy": lambda pred, target: {"cross_entropy": cross_entropy(pred, target)},
    "ce": lambda pred, target: {"cross_entropy": cross_entropy(pred, target)},
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

    _ensure_builtin_tasks_registered()
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


def loss_name(config: dict[str, Any]) -> str:
    """Resolve the configured training loss name."""

    value = config.get("training", {}).get("loss")
    if value is not None:
        return str(value).lower()
    return get_task(config).default_loss


def create_loss(config: dict[str, Any]) -> LossFn:
    """Build the configured loss function from the registry."""

    task = get_task(config)
    name = loss_name(config)
    if name == "task_default" and task.create_loss is not None:
        return task.create_loss(config)
    if task.create_loss is not None and config.get("training", {}).get("loss") is None:
        return task.create_loss(config)
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown training loss: {name}")
    return LOSS_REGISTRY[name](config)


def metric_names(config: dict[str, Any]) -> tuple[str, ...]:
    """Resolve the configured evaluation metric names.

    Configured metrics are merged with the task defaults so task-specific
    reporting remains available even when callers request a subset.
    """

    task = get_task(config)
    configured = config.get("evaluation", {}).get("metrics")
    names: list[str] = []
    if configured is None:
        names.extend(task.default_metrics)
    elif isinstance(configured, str):
        names.append(str(configured).lower())
    else:
        names.extend(str(name).lower() for name in configured)
    for name in task.default_metrics:
        if name not in names:
            names.append(name)
    primary = task.primary_metric
    if primary not in names:
        names.append(primary)
    return tuple(names)


def compute_metrics(config: dict[str, Any], pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute configured metrics from the registry."""

    task = get_task(config)
    configured = config.get("evaluation", {}).get("metrics")
    if task.compute_metrics is not None:
        task_result = task.compute_metrics(pred, target)
        if configured is None:
            return task_result
        requested_names = list(metric_names(config))
        result: dict[str, float] = {}
        for name in requested_names:
            if name in task_result:
                result[name] = task_result[name]
                continue
            if name not in METRIC_REGISTRY:
                raise ValueError(f"Unknown evaluation metric: {name}")
            result.update(METRIC_REGISTRY[name](pred, target))
        for compatibility_name in ("mse", "mae", "mape"):
            if compatibility_name in task_result and compatibility_name not in result:
                result[compatibility_name] = task_result[compatibility_name]
        return result
    result: dict[str, float] = {}
    for name in metric_names(config):
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown evaluation metric: {name}")
        result.update(METRIC_REGISTRY[name](pred, target))
    return result


def optimizer_name(config: dict[str, Any]) -> str:
    """Resolve the configured optimizer name."""

    value = config.get("training", {}).get("optimizer")
    if value is not None:
        return str(value).lower()
    return get_task(config).default_optimizer


def build_optimizer(parameters: Iterable[torch.nn.Parameter], config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the configured optimizer from the registry."""

    training_cfg = config.get("training", {})
    name = optimizer_name(config)
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    if name == "adam":
        eps = float(training_cfg.get("optimizer_eps", 1e-8))
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, eps=eps)
    if name == "adamw":
        eps = float(training_cfg.get("optimizer_eps", 1e-8))
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, eps=eps)
    if name == "sgd":
        momentum = float(training_cfg.get("momentum", 0.0))
        nesterov = bool(training_cfg.get("nesterov", False))
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    raise ValueError(f"Unsupported training optimizer: {name}")


def annotate_model(model: nn.Module, config: dict[str, Any]) -> nn.Module:
    """Attach resolved task metadata to a built model for generic loops."""

    task = get_task(config)
    setattr(model, "_federated_task_type", task.name)
    setattr(model, "_federated_task_config", config)
    return model


def get_model_task(model: nn.Module) -> TaskSpec:
    """Resolve the task plugin attached to a model, falling back to forecasting."""

    _ensure_builtin_tasks_registered()
    name = getattr(model, "_federated_task_type", "forecasting")
    if name not in _TASKS:
        raise ValueError(f"Unknown model task type: {name}")
    return _TASKS[name]


def get_model_config(model: nn.Module) -> dict[str, Any]:
    """Return the config attached to a model if available."""

    return getattr(model, "_federated_task_config", {})
