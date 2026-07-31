"""Model parameter serialization, aggregation, sparse compression, and DP noise."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch


StateDict = OrderedDict[str, torch.Tensor]


@dataclass
class SparseUpdate:
    """Top-k sparse tensor update used to reduce communication volume."""

    names: list[str]
    shapes: list[tuple[int, ...]]
    indices: torch.Tensor
    values: torch.Tensor
    total_numel: int

    @property
    def nbytes(self) -> int:
        """Return serialized sparse payload size in bytes."""

        return self.indices.numel() * self.indices.element_size() + self.values.numel() * self.values.element_size()


def serialize_model(model: torch.nn.Module) -> StateDict:
    """Return an ordered CPU clone of model parameters and buffers."""

    return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in model.state_dict().items())


def load_serialized(model: torch.nn.Module, state: StateDict, device: torch.device | str = "cpu") -> None:
    """Load a serialized state dict onto a model."""

    model.load_state_dict(OrderedDict((name, tensor.to(device)) for name, tensor in state.items()))


def subtract_state(new: StateDict, old: StateDict) -> StateDict:
    """Return ``new - old`` for every tensor in a state dict.

    Example:
        Use this to build a client update before sparse compression.
    """

    return OrderedDict((name, new[name] - old[name]) for name in old.keys())


def add_update(state: StateDict, update: StateDict, scale: float = 1.0) -> StateDict:
    """Apply an update to a state dict.

    Example:
        ``next_state = add_update(global_state, averaged_update)``.
    """

    return OrderedDict((name, state[name] + update[name] * scale) for name in state.keys())


def average_states(states: Iterable[StateDict], weights: Iterable[float] | None = None) -> StateDict:
    """Weighted average dense model states for standard FedAvg.

    Example:
        ``average_states(client_states, weights=[10, 20, 30])``.
    """

    states = list(states)
    if not states:
        raise ValueError("Cannot average an empty state list")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    weights = list(weights)
    total = sum(weights)
    weights = [w / total for w in weights]
    result = OrderedDict()
    for name in states[0].keys():
        result[name] = sum(state[name] * weight for state, weight in zip(states, weights))
    return result


def state_num_bytes(state: StateDict) -> int:
    """Return the dense serialized size of a state dict in bytes."""

    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def state_num_parameters(state: StateDict) -> int:
    """Return the number of scalar values carried by a serialized state."""

    return sum(tensor.numel() for tensor in state.values())


def _flatten_state(update: StateDict) -> tuple[list[str], list[tuple[int, ...]], torch.Tensor]:
    """Flatten a state dict while preserving names and original tensor shapes."""

    names = list(update.keys())
    shapes = [tuple(update[name].shape) for name in names]
    flat = torch.cat([update[name].reshape(-1).cpu() for name in names])
    return names, shapes, flat


def _validate_fraction(fraction: float) -> None:
    """Validate sparse compression fraction."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")


def compress_topk(update: StateDict, fraction: float) -> SparseUpdate:
    """Compress an update with global top-k magnitude sparsification.

    Example:
        ``compress_topk(update, 0.05)`` keeps the largest five percent of
        update coordinates and drops the rest.
    """

    _validate_fraction(fraction)
    names, shapes, flat = _flatten_state(update)
    total = flat.numel()
    k = max(1, int(total * fraction))
    _, indices = torch.topk(flat.abs(), k)
    values = flat[indices].to(torch.float32)
    return SparseUpdate(names, shapes, indices.to(torch.int64), values, total)


def compress_randomk(update: StateDict, fraction: float, generator: torch.Generator | None = None) -> SparseUpdate:
    """Compress an update with unbiased random-k sparsification.

    This is the random-k compressor commonly used by compressed SGD variants:
    selected coordinates are scaled by ``d / k`` so the decompressed sparse
    vector is an unbiased estimator of the dense update.

    Example:
        ``compress_randomk(update, 0.05)`` randomly uploads five percent of
        coordinates with unbiased scaling.
    """

    _validate_fraction(fraction)
    names, shapes, flat = _flatten_state(update)
    total = flat.numel()
    k = max(1, int(total * fraction))
    indices = torch.randperm(total, generator=generator)[:k]
    scale = float(total) / float(k)
    values = flat[indices].to(torch.float32) * scale
    return SparseUpdate(names, shapes, indices.to(torch.int64), values, total)


def decompress_topk(sparse: SparseUpdate) -> StateDict:
    """Reconstruct a dense update state from a sparse payload.

    Example:
        ``dense_update = decompress_topk(compress_topk(update, 0.05))``.
    """

    flat = torch.zeros(sparse.total_numel, dtype=sparse.values.dtype)
    flat[sparse.indices] = sparse.values
    result = OrderedDict()
    offset = 0
    for name, shape in zip(sparse.names, sparse.shapes):
        numel = int(torch.tensor(shape).prod().item()) if shape else 1
        result[name] = flat[offset : offset + numel].reshape(shape)
        offset += numel
    return result



def clip_state_update(update: StateDict, clip_norm: float) -> StateDict:
    """Clip a dense update by global L2 norm.

    Example:
        ``clip_state_update(update, 1.0)`` rescales all tensors together when
        the concatenated update norm exceeds one.
    """

    if clip_norm <= 0:
        return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in update.items())
    _, _, flat = _flatten_state(update)
    norm = torch.linalg.vector_norm(flat)
    scale = min(1.0, float(clip_norm / (norm + 1e-12)))
    return OrderedDict((name, tensor.detach().cpu().clone() * scale) for name, tensor in update.items())


def privatize_state_update(
    update: StateDict,
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> StateDict:
    """Apply local Gaussian DP perturbation to a dense update.

    The full update is clipped first, then each coordinate receives Gaussian
    noise with standard deviation ``noise_multiplier * clip_norm``.

    Example:
        ``private = privatize_state_update(update, 1.0, 0.05)`` prepares a
        noisy update before sparse communication.
    """

    clipped = clip_state_update(update, clip_norm)
    if noise_multiplier <= 0 or clip_norm <= 0:
        return clipped
    std = float(noise_multiplier) * float(clip_norm)
    private = OrderedDict()
    for name, tensor in clipped.items():
        noise = torch.randn(tensor.shape, dtype=tensor.dtype, generator=generator) * std
        private[name] = tensor + noise
    return private


def privatize_sparse_update(sparse: SparseUpdate, clip_norm: float, noise_multiplier: float) -> SparseUpdate:
    """Clip and perturb sparse values with Gaussian local-DP style noise.

    Example:
        ``privatize_sparse_update(sparse, clip_norm=1.0, noise_multiplier=0.5)``
        keeps the sparse support and sends noisy clipped values.
    """

    values = sparse.values.clone()
    norm = torch.linalg.vector_norm(values)
    if clip_norm > 0 and norm > clip_norm:
        values = values * (clip_norm / (norm + 1e-12))
    if noise_multiplier > 0 and clip_norm > 0:
        values = values + torch.randn_like(values) * (noise_multiplier * clip_norm)
    return SparseUpdate(sparse.names, sparse.shapes, sparse.indices.clone(), values, sparse.total_numel)
