"""Local training and evaluation routines."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from federated_ts.metrics import mae, mape, mse


def train_one_epoch(model: nn.Module, loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train one local epoch and return sample-weighted MSE loss.

    Example:
        ``loss = train_one_epoch(model, train_loader, optimizer, device)``.
    """

    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
        total += x.shape[0]
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> dict[str, float]:
    """Evaluate a forecasting model and return MSE, MAE, and MAPE.

    Example:
        ``metrics = evaluate(model, val_loader, torch.device("cpu"))``.
    """

    model.eval()
    preds = []
    targets = []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu())
        targets.append(y.cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    return {"mse": mse(pred, target), "mae": mae(pred, target), "mape": mape(pred, target)}


def first_batch_gradient(model: nn.Module, loader: Iterable, device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return gradients for the first batch, useful for reconstruction attacks."""

    model.train()
    criterion = nn.MSELoss()
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    model.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=False)
    return [grad.detach().cpu() for grad in grads], x.detach().cpu(), y.detach().cpu()

