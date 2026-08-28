"""Helpers for deterministic CPU/CUDA seeding."""

from __future__ import annotations

from typing import Any

import torch


def seed_cuda_device(seed: int, device: torch.device | str | None) -> None:
    """Seed only one selected CUDA device without touching all visible GPUs."""

    if device is None or not torch.cuda.is_available():
        return
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return
    with torch.cuda.device(resolved):
        torch.cuda.manual_seed(int(seed))
