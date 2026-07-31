"""Forecasting metrics."""

from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean squared error.

    Example:
        ``mse(torch.tensor([1.0]), torch.tensor([2.0])) == 1.0``.
    """

    return torch.mean((pred - target) ** 2).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean absolute error.

    Example:
        ``mae(torch.tensor([1.0]), torch.tensor([2.0])) == 1.0``.
    """

    return torch.mean(torch.abs(pred - target)).item()


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Return mean absolute percentage error in percent.

    Example:
        ``mape(torch.tensor([2.0]), torch.tensor([1.0])) == 100.0``.
    """

    denom = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(torch.abs((pred - target) / denom)).item() * 100

