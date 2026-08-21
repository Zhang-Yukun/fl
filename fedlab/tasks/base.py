"""Task abstraction shared by federated training backends.

Example:
    ``TaskSpec`` lets the federated core ask a task plugin for its model,
    data, loss, and metric behavior without importing task-specific modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], dict[str, float]]


@dataclass(frozen=True)
class TaskSpec:
    """Concrete hooks required by the task-agnostic federated core."""

    name: str
    build_model: Callable[[dict[str, Any]], nn.Module]
    build_federated_loaders: Callable[[dict[str, Any]], tuple[dict[str, Any], Any, Any]]
    create_loss: Callable[[dict[str, Any]], LossFn] | None = None
    compute_metrics: MetricFn | None = None
    default_loss: str = "mse"
    default_metrics: tuple[str, ...] = ("mse",)
    default_optimizer: str = "sgd"
    primary_metric: str = "mse"
    primary_metric_mode: str = "min"
