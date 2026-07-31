"""Model parameter serialization, aggregation, and sparse update compression."""

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
        return self.indices.numel() * self.indices.element_size() + self.values.numel() * self.values.element_size()


def serialize_model(model: torch.nn.Module) -> StateDict:
    """Return an ordered CPU clone of model parameters and buffers."""

    return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in model.state_dict().items())


def load_serialized(model: torch.nn.Module, state: StateDict, device: torch.device | str = "cpu") -> None:
    """Load a serialized state dict onto a model."""

    model.load_state_dict(OrderedDict((name, tensor.to(device)) for name, tensor in state.items()))


def subtract_state(new: StateDict, old: StateDict) -> StateDict:
    return OrderedDict((name, new[name] - old[name]) for name in old.keys())


def add_update(state: StateDict, update: StateDict, scale: float = 1.0) -> StateDict:
    return OrderedDict((name, state[name] + update[name] * scale) for name in state.keys())


def average_states(states: Iterable[StateDict], weights: Iterable[float] | None = None) -> StateDict:
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
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def state_num_parameters(state: StateDict) -> int:
    """Return the number of scalar values carried by a serialized state."""

    return sum(tensor.numel() for tensor in state.values())


def compress_topk(update: StateDict, fraction: float) -> SparseUpdate:
    """Compress an update with global top-k magnitude sparsification."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    names = list(update.keys())
    shapes = [tuple(update[name].shape) for name in names]
    flat = torch.cat([update[name].reshape(-1).cpu() for name in names])
    total = flat.numel()
    k = max(1, int(total * fraction))
    _, indices = torch.topk(flat.abs(), k)
    values = flat[indices].to(torch.float32)
    return SparseUpdate(names, shapes, indices.to(torch.int64), values, total)


def decompress_topk(sparse: SparseUpdate) -> StateDict:
    flat = torch.zeros(sparse.total_numel, dtype=sparse.values.dtype)
    flat[sparse.indices] = sparse.values
    result = OrderedDict()
    offset = 0
    for name, shape in zip(sparse.names, sparse.shapes):
        numel = int(torch.tensor(shape).prod().item()) if shape else 1
        result[name] = flat[offset : offset + numel].reshape(shape)
        offset += numel
    return result

