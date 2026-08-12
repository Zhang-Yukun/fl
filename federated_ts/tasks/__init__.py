"""Task registry exposed to the rest of the framework."""

from federated_ts.tasks.base import TaskSpec
from federated_ts.tasks.registry import (
    annotate_model,
    get_model_config,
    get_model_task,
    get_task,
    primary_metric,
    primary_metric_mode,
    task_type,
)

__all__ = [
    "TaskSpec",
    "annotate_model",
    "get_model_config",
    "get_model_task",
    "get_task",
    "primary_metric",
    "primary_metric_mode",
    "task_type",
]
