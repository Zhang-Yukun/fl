"""Adaptive clipping and clipping-aware RDP accounting utilities.

Example:
    ``threshold = adaptive_clip_threshold([0.5, 1.0, 2.0], 1.2, 0.1, 5.0)``
    computes the median-based clipping threshold from the paper.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch

from fedlab.utils.serialization import StateDict


def state_l2_norm(state: StateDict) -> float:
    """Return the global L2 norm of a serialized model update.

    Example:
        ``norm = state_l2_norm(update_state)`` computes ``||Delta||_2``.
    """

    if not state:
        return 0.0
    flat = torch.cat([tensor.detach().reshape(-1).cpu() for tensor in state.values()])
    return float(torch.linalg.vector_norm(flat).item())


def adaptive_clip_threshold(
    norms: Iterable[float],
    clip_factor: float,
    min_clip: float,
    max_clip: float,
) -> tuple[float, float, float]:
    """Compute the paper's median-based adaptive clipping threshold.

    Returns ``(median_norm, raw_clip, clipped_threshold)``.

    Example:
        ``median_norm, raw_clip, clip_t = adaptive_clip_threshold([1.0, 2.0], 1.5, 0.5, 5.0)``
    """

    values = sorted(float(value) for value in norms)
    if not values:
        raise ValueError("norms must not be empty")
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        median_norm = values[midpoint]
    else:
        median_norm = 0.5 * (values[midpoint - 1] + values[midpoint])
    raw_clip = float(clip_factor) * float(median_norm)
    clipped_threshold = min(max(float(raw_clip), float(min_clip)), float(max_clip))
    return float(median_norm), float(raw_clip), float(clipped_threshold)


def clip_state_to_norm(update: StateDict, clip_norm: float) -> tuple[StateDict, float, float]:
    """Clip a dense update to a target L2 norm and return scaling metadata.

    Returns ``(clipped_update, original_norm, applied_scale)``.

    Example:
        ``clipped, norm, scale = clip_state_to_norm(update, 1.0)``.
    """

    original_norm = state_l2_norm(update)
    if clip_norm <= 0:
        return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in update.items()), original_norm, 1.0
    scale = min(1.0, float(clip_norm / (original_norm + 1e-12)))
    clipped = OrderedDict((name, tensor.detach().cpu().clone() * scale) for name, tensor in update.items())
    return clipped, original_norm, scale


def add_gaussian_noise(
    update: StateDict,
    noise_std: float,
    generator: torch.Generator | None = None,
) -> StateDict:
    """Add isotropic Gaussian noise to every coordinate of a dense update.

    Example:
        ``private = add_gaussian_noise(update, noise_std=0.05)``.
    """

    if noise_std <= 0:
        return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in update.items())
    noisy = OrderedDict()
    for name, tensor in update.items():
        noise = torch.randn(tensor.shape, dtype=tensor.dtype, generator=generator) * float(noise_std)
        noisy[name] = tensor.detach().cpu().clone() + noise
    return noisy


@dataclass
class AdaptiveRdpStep:
    """One round of clipping-aware RDP accounting state."""

    round_index: int
    sampling_rate: float
    rdp_alpha: float
    delta: float
    reference_clip_norm: float
    adaptive_clip_norm: float
    median_update_norm: float
    raw_clip_norm: float
    noise_multiplier: float
    noise_std: float
    round_rdp: float
    total_rdp: float
    epsilon: float


class AdaptiveClippedRdpAccountant:
    """Clipping-schedule-aware RDP accountant from the paper.

    Example:
        ``accountant = AdaptiveClippedRdpAccountant(16.0, 1e-5, 2.0)``
        followed by ``accountant.step(...)`` tracks cumulative privacy loss.
    """

    def __init__(self, rdp_alpha: float, delta: float, noise_multiplier: float):
        """Create an accountant with fixed ``alpha``, ``delta``, and ``sigma``."""

        if rdp_alpha <= 1:
            raise ValueError("rdp_alpha must be > 1")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        self.rdp_alpha = float(rdp_alpha)
        self.delta = float(delta)
        self.noise_multiplier = float(noise_multiplier)
        self.total_rdp = 0.0

    def step(
        self,
        round_index: int,
        sampling_rate: float,
        adaptive_clip_norm: float,
        reference_clip_norm: float,
        median_update_norm: float,
        raw_clip_norm: float,
    ) -> AdaptiveRdpStep:
        """Accumulate one clipping-aware RDP contribution.

        Example:
            ``state = accountant.step(0, 1.0, 0.8, 1.0, 0.7, 0.84)``.
        """

        sampling_rate = float(sampling_rate)
        adaptive_clip_norm = float(adaptive_clip_norm)
        reference_clip_norm = float(reference_clip_norm)
        if self.noise_multiplier <= 0 or reference_clip_norm <= 0:
            round_rdp = float("inf")
            self.total_rdp = float("inf")
            epsilon = float("inf")
            noise_std = max(self.noise_multiplier, 0.0) * max(reference_clip_norm, 0.0)
        else:
            ratio = adaptive_clip_norm / reference_clip_norm
            round_rdp = (sampling_rate ** 2) * self.rdp_alpha / (2.0 * (self.noise_multiplier ** 2)) * (ratio ** 2)
            self.total_rdp += round_rdp
            epsilon = self.total_rdp + torch.log(torch.tensor(1.0 / self.delta)).item() / (self.rdp_alpha - 1.0)
            noise_std = self.noise_multiplier * reference_clip_norm
        return AdaptiveRdpStep(
            round_index=int(round_index),
            sampling_rate=sampling_rate,
            rdp_alpha=self.rdp_alpha,
            delta=self.delta,
            reference_clip_norm=reference_clip_norm,
            adaptive_clip_norm=adaptive_clip_norm,
            median_update_norm=float(median_update_norm),
            raw_clip_norm=float(raw_clip_norm),
            noise_multiplier=self.noise_multiplier,
            noise_std=float(noise_std),
            round_rdp=float(round_rdp),
            total_rdp=float(self.total_rdp),
            epsilon=float(epsilon),
        )
