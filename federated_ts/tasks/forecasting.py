"""Forecasting task plugin backed by the existing rare-earth pipeline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.modeling.forecasting import build_model as build_forecasting_model
from federated_ts.tasks.base import TaskSpec
from federated_ts.utils.metrics import mae, mape, mse


def _create_loss(_config: dict[str, Any]):
    """Return the forecasting objective used by the existing code path."""

    return nn.MSELoss()


def _compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute forecasting metrics on batched model outputs."""

    return {"mse": mse(pred, target), "mae": mae(pred, target), "mape": mape(pred, target)}


FORECASTING_TASK = TaskSpec(
    name="forecasting",
    build_model=build_forecasting_model,
    build_federated_loaders=build_federated_loaders,
    create_loss=_create_loss,
    compute_metrics=_compute_metrics,
    primary_metric="mse",
    primary_metric_mode="min",
)
