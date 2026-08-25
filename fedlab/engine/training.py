"""Local training and evaluation routines."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from fedlab.tasks.registry import build_optimizer, compute_metrics as compute_registry_metrics, create_loss as create_registry_loss, get_model_config


def _prediction(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run the model forward pass for one minibatch."""

    return model(x)


def _loss_fn(model: nn.Module):
    """Resolve the configured task-aware loss function for a model."""

    try:
        return create_registry_loss(get_model_config(model))
    except Exception:
        return nn.MSELoss()


def _metric_fn(model: nn.Module):
    """Resolve the configured task-aware metric function for a model."""

    try:
        config = get_model_config(model)
        return lambda pred, target: compute_registry_metrics(config, pred, target)
    except Exception:
        return lambda pred, target: {"mse": torch.mean((pred - target) ** 2).item()}


def build_training_optimizer(parameters, config: dict) -> torch.optim.Optimizer:
    """Build the configured training optimizer through the registry."""

    return build_optimizer(parameters, config)


def train_one_epoch(model: nn.Module, loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Train one local epoch and return sample-weighted task loss.

    Example:
        ``loss = train_one_epoch(model, train_loader, optimizer, device)``.
    """

    model.train()
    criterion = _loss_fn(model)
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(_prediction(model, x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
        total += x.shape[0]
    return total_loss / max(total, 1)


def train_n_steps(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
) -> float:
    """Train for a fixed number of local optimizer steps.

    Example:
        ``loss = train_n_steps(model, train_loader, optimizer, device, steps=5)``
        performs exactly five minibatch updates, cycling the loader if needed.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")

    model.train()
    criterion = _loss_fn(model)
    total_loss = 0.0
    total = 0
    completed = 0
    iterator = iter(loader)

    while completed < steps:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(_prediction(model, x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.shape[0]
        total += x.shape[0]
        completed += 1
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device) -> dict[str, float]:
    """Evaluate a model and return task metrics.

    Example:
        ``metrics = evaluate(model, val_loader, torch.device("cpu"))``.
    """

    model.eval()
    preds = []
    targets = []
    for x, y in loader:
        preds.append(_prediction(model, x.to(device)).cpu())
        targets.append(y.cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    return _metric_fn(model)(pred, target)


@torch.no_grad()
def predict_first_batch(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    max_samples: int | None = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one input batch, its prediction, and the ground-truth target."""

    model.eval()
    x, y = next(iter(loader))
    if max_samples is not None:
        x = x[:max_samples]
        y = y[:max_samples]
    prediction = _prediction(model, x.to(device)).cpu()
    return x.cpu(), prediction, y.cpu()


@torch.no_grad()
def predict_first_batch_for_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    loader: Iterable,
    device: torch.device,
    max_samples: int | None = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a specific state into ``model`` and predict the first batch.

    Example:
        ``predict_first_batch_for_state(model, global_state, val_loader, device)``
        renders protocol/oracle predictions from explicitly selected states.
    """

    model.load_state_dict(state)
    return predict_first_batch(model, loader, device, max_samples=max_samples)


def _select_loader_samples(
    loader: Iterable,
    max_samples: int | None = None,
    batch_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one selected batch or a concatenated prefix of subsequent batches."""

    iterator = iter(loader)
    x = y = None
    for _ in range(max(0, int(batch_index)) + 1):
        x, y = next(iterator)
    if max_samples is None:
        return x, y
    target_count = max(1, int(max_samples))
    x_parts = [x]
    y_parts = [y]
    total = int(x.shape[0])
    while total < target_count:
        try:
            next_x, next_y = next(iterator)
        except StopIteration:
            break
        x_parts.append(next_x)
        y_parts.append(next_y)
        total += int(next_x.shape[0])
    merged_x = torch.cat(x_parts, dim=0)
    merged_y = torch.cat(y_parts, dim=0)
    return merged_x[:target_count], merged_y[:target_count]


def first_batch_sample(
    loader: Iterable,
    device: torch.device,
    max_samples: int | None = None,
    batch_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one selected batch without computing gradients.

    Example:
        ``x, y = first_batch_sample(loader, device, max_samples=1, batch_index=2)``
        fetches the third batch for payload-based inversion experiments.
    """

    x, y = _select_loader_samples(loader, max_samples=max_samples, batch_index=batch_index)
    return x.to(device), y.to(device)


def first_batch_gradient(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    max_samples: int | None = None,
    model_mode: str = "train",
    batch_index: int = 0,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return gradients for a selected batch, useful for reconstruction attacks.

    Example:
        ``first_batch_gradient(model, loader, device, max_samples=1, model_mode="eval", batch_index=2)``
        attacks the third batch from the current loader order.
    """

    if model_mode == "eval":
        model.eval()
    else:
        model.train()
    criterion = _loss_fn(model)
    x, y = _select_loader_samples(loader, max_samples=max_samples, batch_index=batch_index)
    x = x.to(device)
    y = y.to(device)
    model.zero_grad(set_to_none=True)
    loss = criterion(_prediction(model, x), y)
    trainable_parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    grads = torch.autograd.grad(loss, trainable_parameters, create_graph=False)
    return [grad.detach().cpu() for grad in grads], x.detach().cpu(), y.detach().cpu()
