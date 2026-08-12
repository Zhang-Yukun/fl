"""Model parameter serialization, aggregation, sparse compression, and DP noise."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch


StateDict = OrderedDict[str, torch.Tensor]
QUANT_SCALE_SUFFIX = ".__scale__"


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
    """Load a serialized state dict onto a model.

    The loader accepts both current checkpoints and legacy PatchTST checkpoints
    whose parameter names were prefixed with ``model.`` by an old wrapper.
    """

    device_state = OrderedDict((name, tensor.to(device)) for name, tensor in state.items())
    expected_keys = tuple(model.state_dict().keys())
    if expected_keys and all(key.startswith("model.") for key in device_state.keys()) and not expected_keys[0].startswith("model."):
        device_state = OrderedDict((name.removeprefix("model."), tensor) for name, tensor in device_state.items())
    model.load_state_dict(device_state)


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


def quantize_state_update(
    update: StateDict,
    dtype: str = "float16",
    stochastic_rounding: bool = False,
    generator: torch.Generator | None = None,
) -> StateDict:
    """Quantize a dense update tensor-by-tensor for communication reduction.

    Example:
        ``quantize_state_update(update, dtype="float16")`` halves the
        upload payload while keeping a dense update structure.

        ``quantize_state_update(update, dtype="int8", stochastic_rounding=True)``
        performs per-tensor absmax normalization, applies randomized rounding,
        sends one float scale per tensor, and stores the normalized values as int8.
    """

    normalized_dtype = str(dtype).lower()
    quantized_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(normalized_dtype)
    if quantized_dtype is not None:
        return OrderedDict((name, tensor.detach().cpu().clone().to(quantized_dtype)) for name, tensor in update.items())
    if normalized_dtype in {"sign", "signum", "onebit", "1bit"}:
        quantized: StateDict = OrderedDict()
        for name, tensor in update.items():
            base = tensor.detach().cpu().clone().to(torch.float32)
            scale = max(float(base.abs().mean().item()), 1e-12)
            signs = torch.sign(base).to(torch.int8)
            quantized[name] = signs
            quantized[f"{name}{QUANT_SCALE_SUFFIX}"] = torch.tensor(scale, dtype=torch.float32)
        return quantized
    if normalized_dtype not in {"int8", "qint8", "absmax_int8", "scaled_int8"}:
        raise ValueError(f"Unsupported quantization dtype: {dtype}")

    quantized: StateDict = OrderedDict()
    for name, tensor in update.items():
        base = tensor.detach().cpu().clone().to(torch.float32)
        max_abs = float(base.abs().max().item())
        scale = max(max_abs / 127.0, 1e-12)
        normalized = torch.clamp(base / scale, -127.0, 127.0)
        if stochastic_rounding:
            lower = torch.floor(normalized)
            probability = normalized - lower
            random = torch.rand(normalized.shape, generator=generator, dtype=torch.float32)
            rounded = lower + (random < probability).to(torch.float32)
        else:
            rounded = torch.round(normalized)
        q = torch.clamp(rounded, -127.0, 127.0).to(torch.int8)
        quantized[name] = q
        quantized[f"{name}{QUANT_SCALE_SUFFIX}"] = torch.tensor(scale, dtype=torch.float32)
    return quantized


def dequantize_state_update(update: StateDict) -> StateDict:
    """Restore a quantized dense update to float32 tensors for aggregation.

    Example:
        ``dequantize_state_update(quantized_update)`` prepares uploaded values
        for server-side FedAvg aggregation.
    """

    restored: StateDict = OrderedDict()
    for name, tensor in update.items():
        if name.endswith(QUANT_SCALE_SUFFIX):
            continue
        scale_name = f"{name}{QUANT_SCALE_SUFFIX}"
        if scale_name in update:
            scale = float(update[scale_name].detach().cpu().item())
            restored[name] = tensor.detach().cpu().to(torch.float32) * scale
        else:
            restored[name] = tensor.detach().cpu().clone().to(torch.float32)
    return restored


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


def quantize_qsgd_state_update(
    update: StateDict,
    levels: int,
    generator: torch.Generator | None = None,
) -> StateDict:
    """Quantize a dense update with QSGD-style stochastic level coding."""

    if levels <= 0:
        raise ValueError("levels must be positive")
    quantized: StateDict = OrderedDict()
    for name, tensor in update.items():
        base = tensor.detach().cpu().clone().to(torch.float32)
        norm = float(torch.linalg.vector_norm(base.reshape(-1)).item())
        scale = max(norm, 1e-12)
        scaled = torch.clamp(base.abs() / scale, 0.0, 1.0) * float(levels)
        lower = torch.floor(scaled)
        probability = scaled - lower
        if generator is not None:
            random = torch.rand(scaled.shape, generator=generator, dtype=torch.float32)
        else:
            random = torch.rand(scaled.shape, dtype=torch.float32)
        bucket = torch.clamp(lower + (random < probability).to(torch.float32), 0.0, float(levels))
        signed_bucket = bucket * torch.sign(base)
        quantized[name] = signed_bucket.to(torch.int16)
        quantized[f"{name}{QUANT_SCALE_SUFFIX}"] = torch.tensor(scale, dtype=torch.float32)
    return quantized


def dequantize_qsgd_state_update(update: StateDict, levels: int) -> StateDict:
    """Restore a QSGD-style quantized update to float32 tensors."""

    if levels <= 0:
        raise ValueError("levels must be positive")
    restored: StateDict = OrderedDict()
    for name, tensor in update.items():
        if name.endswith(QUANT_SCALE_SUFFIX):
            continue
        scale_name = f"{name}{QUANT_SCALE_SUFFIX}"
        scale = float(update[scale_name].detach().cpu().item()) if scale_name in update else 1.0
        restored[name] = tensor.detach().cpu().to(torch.float32) * (scale / float(levels))
    return restored
