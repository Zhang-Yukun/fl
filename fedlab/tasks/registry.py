"""Task registry for task-agnostic model, data, loss, metric, and optimizer resolution."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import nn

from fedlab.tasks.base import LossFn, MetricFn, TaskSpec
from fedlab.utils.metrics import accuracy, cross_entropy, mae, mape, mse


LossBuilder = Callable[[dict[str, Any]], LossFn]

_TASKS: dict[str, TaskSpec] = {}
_BUILTIN_TASKS_LOADED = False
_LOSS_BUILDERS: dict[str, LossBuilder] = {}
_METRIC_BUILDERS: dict[str, MetricFn] = {}
LOSS_REGISTRY = _LOSS_BUILDERS
METRIC_REGISTRY = _METRIC_BUILDERS


def _normalize_name(name: str) -> str:
    """Normalize registry keys to the canonical lowercase form."""

    return str(name).strip().lower()


def _register_named_plugin(registry: dict[str, Any], kind: str, plugin: Any, primary_name: str, aliases: tuple[str, ...], replace: bool = False) -> Any:
    """Register one named plugin and optional aliases into one registry."""

    names = (_normalize_name(primary_name), *(_normalize_name(alias) for alias in aliases))
    for name in names:
        existing = registry.get(name)
        if existing is not None and existing is not plugin and not replace:
            raise ValueError(f"{kind} alias already registered: {name}")
    for name in names:
        registry[name] = plugin
    return plugin


def register_task(task: TaskSpec, *aliases: str, replace: bool = False) -> TaskSpec:
    """Register one task plugin and optional aliases."""

    return _register_named_plugin(_TASKS, 'Task', task, task.name, aliases, replace=replace)


def task_plugin(*aliases: str, replace: bool = False):
    """Decorator for registering one ``TaskSpec`` as a task plugin."""

    def decorator(task: TaskSpec) -> TaskSpec:
        return register_task(task, *aliases, replace=replace)

    return decorator


def register_loss(name: str, builder: LossBuilder, *aliases: str, replace: bool = False) -> LossBuilder:
    """Register one loss builder and optional aliases."""

    return _register_named_plugin(_LOSS_BUILDERS, 'Loss', builder, name, aliases, replace=replace)


def loss_plugin(name: str, *aliases: str, replace: bool = False):
    """Decorator for registering one loss builder."""

    def decorator(builder: LossBuilder) -> LossBuilder:
        return register_loss(name, builder, *aliases, replace=replace)

    return decorator


def register_metric(name: str, metric: MetricFn, *aliases: str, replace: bool = False) -> MetricFn:
    """Register one metric builder and optional aliases."""

    return _register_named_plugin(_METRIC_BUILDERS, 'Metric', metric, name, aliases, replace=replace)


def metric_plugin(name: str, *aliases: str, replace: bool = False):
    """Decorator for registering one metric builder."""

    def decorator(metric: MetricFn) -> MetricFn:
        return register_metric(name, metric, *aliases, replace=replace)

    return decorator


def list_registered_tasks() -> dict[str, TaskSpec]:
    """Return a snapshot of currently registered tasks."""

    _ensure_builtin_tasks_registered()
    return dict(_TASKS)


def list_registered_losses() -> dict[str, LossBuilder]:
    """Return a snapshot of currently registered losses."""

    return dict(_LOSS_BUILDERS)


def list_registered_metrics() -> dict[str, MetricFn]:
    """Return a snapshot of currently registered metrics."""

    return dict(_METRIC_BUILDERS)


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


def _metric_mse(pred, target):
    return {'mse': mse(pred, target)}


def _metric_mae(pred, target):
    return {'mae': mae(pred, target)}


def _metric_mape(pred, target):
    return {'mape': mape(pred, target)}


def _metric_accuracy(pred, target):
    return {'accuracy': accuracy(pred, target)}


def _metric_cross_entropy(pred, target):
    return {'cross_entropy': cross_entropy(pred, target)}


register_loss('mse', _loss_mse, 'l2')
register_loss('mae', _loss_l1, 'l1')
register_loss('smooth_l1', _loss_smooth_l1)
register_loss('huber', _loss_huber)
register_loss('cross_entropy', _loss_cross_entropy, 'ce')
register_metric('mse', _metric_mse)
register_metric('mae', _metric_mae)
register_metric('mape', _metric_mape)
register_metric('accuracy', _metric_accuracy)
register_metric('cross_entropy', _metric_cross_entropy, 'ce')


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
    if name not in _LOSS_BUILDERS:
        raise ValueError(f"Unknown training loss: {name}")
    return _LOSS_BUILDERS[name](config)


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
            if name not in _METRIC_BUILDERS:
                raise ValueError(f"Unknown evaluation metric: {name}")
            result.update(_METRIC_BUILDERS[name](pred, target))
        for compatibility_name in ("mse", "mae", "mape"):
            if compatibility_name in task_result and compatibility_name not in result:
                result[compatibility_name] = task_result[compatibility_name]
        return result
    result: dict[str, float] = {}
    for name in metric_names(config):
        if name not in _METRIC_BUILDERS:
            raise ValueError(f"Unknown evaluation metric: {name}")
        result.update(_METRIC_BUILDERS[name](pred, target))
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
