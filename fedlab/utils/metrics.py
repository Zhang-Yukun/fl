"""Forecasting and classification metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean squared error."""

    return torch.mean((pred - target) ** 2).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean absolute error."""

    return torch.mean(torch.abs(pred - target)).item()


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Return mean absolute percentage error in percent."""

    denom = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(torch.abs((pred - target) / denom)).item() * 100


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Return classification accuracy in ``[0, 1]``."""

    if logits.ndim < 2:
        raise ValueError('accuracy expects logits with class dimension')
    predicted = torch.argmax(logits, dim=1)
    return torch.mean((predicted == target).to(torch.float32)).item()


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Return mean cross-entropy over class logits and integer labels."""

    return F.cross_entropy(logits, target).item()
