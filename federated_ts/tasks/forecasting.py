"""Forecasting task plugin backed by the existing rare-earth pipeline."""

from __future__ import annotations

from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.modeling.forecasting import build_model as build_forecasting_model
from federated_ts.tasks.base import TaskSpec


FORECASTING_TASK = TaskSpec(
    name="forecasting",
    build_model=build_forecasting_model,
    build_federated_loaders=build_federated_loaders,
    default_loss="mse",
    default_metrics=("mse", "mae", "mape"),
    default_optimizer="adam",
    primary_metric="mse",
    primary_metric_mode="min",
)
