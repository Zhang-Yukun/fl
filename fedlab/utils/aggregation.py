"""Adaptive aggregation helpers for federated optimization."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch

from fedlab.utils.serialization import StateDict


def flatten_state(state: StateDict) -> torch.Tensor:
    """Flatten a state dict into one float vector.

    Example:
        ``flat = flatten_state(update)`` produces one vector suitable for
        adaptive weighting objectives over full model updates.
    """

    return torch.cat([tensor.reshape(-1).to(torch.float32) for tensor in state.values()])


def project_simplex(weights: torch.Tensor) -> torch.Tensor:
    """Project a vector onto the probability simplex."""

    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    sorted_weights, _ = torch.sort(weights, descending=True)
    cumsum = torch.cumsum(sorted_weights, dim=0)
    steps = torch.arange(1, len(weights) + 1, device=weights.device, dtype=weights.dtype)
    support = sorted_weights - (cumsum - 1.0) / steps > 0
    rho = int(torch.nonzero(support, as_tuple=False)[-1].item())
    theta = (cumsum[rho] - 1.0) / float(rho + 1)
    return torch.clamp(weights - theta, min=0.0)


def fedaware_weights(
    updates: Iterable[StateDict],
    sample_weights: Iterable[float],
    alpha: float = 0.5,
    steps: int = 50,
    lr: float = 0.1,
) -> list[float]:
    """Compute FedAWARE-style adaptive aggregation weights.

    The implementation follows the paper's adaptive weighted aggregation
    direction by optimizing a simplex-constrained combination of client
    updates, then blending it with the standard sample-count FedAvg prior.

    Example:
        ``fedaware_weights(updates, [10, 20, 30], alpha=0.5)`` returns
        client weights that sum to one.
    """

    updates = list(updates)
    sample_weights = torch.tensor(list(sample_weights), dtype=torch.float32)
    if not updates:
        raise ValueError("updates must be non-empty")
    if len(updates) != len(sample_weights):
        raise ValueError("updates and sample_weights must have the same length")
    if len(updates) == 1:
        return [1.0]
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    sample_weights = sample_weights / sample_weights.sum()
    flat_updates = torch.stack([flatten_state(update) for update in updates], dim=0)
    gram = flat_updates @ flat_updates.t()
    weights = sample_weights.clone()
    for _ in range(max(1, int(steps))):
        grad = 2.0 * (gram @ weights)
        weights = project_simplex(weights - float(lr) * grad)
    blended = (1.0 - float(alpha)) * sample_weights + float(alpha) * weights
    blended = project_simplex(blended)
    return blended.tolist()

