"""Task registry exposed to the rest of the framework."""

from fedlab.tasks.base import TaskSpec
from fedlab.tasks.registry import (
    annotate_model,
    build_optimizer,
    compute_metrics,
    create_loss,
    get_model_config,
    get_model_task,
    get_task,
    loss_name,
    metric_names,
    optimizer_name,
    primary_metric,
    primary_metric_mode,
    task_type,
)

__all__ = [
    "TaskSpec",
    "annotate_model",
    "build_optimizer",
    "compute_metrics",
    "create_loss",
    "get_model_config",
    "loss_name",
    "metric_names",
    "optimizer_name",
    "get_model_task",
    "get_task",
    "primary_metric",
    "primary_metric_mode",
    "task_type",
]
