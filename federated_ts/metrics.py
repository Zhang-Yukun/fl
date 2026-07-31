"""Forecasting metrics."""

from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    denom = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(torch.abs((pred - target) / denom)).item() * 100

