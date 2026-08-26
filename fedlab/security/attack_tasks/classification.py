"""Classification-specific helpers for reconstruction attacks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fedlab.utils.serialization import StateDict


def is_classification_attack(config: dict[str, Any], real_y: torch.Tensor) -> bool:
    """Return whether the current attack target is a classification batch."""

    task_type = str(config.get("task", {}).get("type", "")).lower()
    if task_type == "classification":
        return True
    return real_y.ndim == 1 and not torch.is_floating_point(real_y)


def _classification_num_classes(
    config: dict[str, Any],
    model: nn.Module | None = None,
    sample_x: torch.Tensor | None = None,
) -> int:
    """Resolve the classification output dimension used by label inference."""

    configured = int(config.get("data", {}).get("num_classes", 0))
    if configured > 0:
        return configured
    if model is not None and sample_x is not None:
        with torch.no_grad():
            return int(model(sample_x).shape[1])
    raise ValueError("Could not infer classification num_classes for iDLG label inference")


def _candidate_label_signal_from_tensor(tensor: torch.Tensor, num_classes: int) -> torch.Tensor | None:
    """Project one parameter-shaped gradient/update tensor onto a class-wise score vector."""

    detached = tensor.detach()
    if detached.ndim == 1 and detached.numel() == num_classes:
        return detached.reshape(num_classes)
    if detached.ndim >= 2 and detached.shape[0] == num_classes:
        return detached.reshape(num_classes, -1).sum(dim=1)
    return None


def infer_classification_label(
    config: dict[str, Any],
    model: nn.Module,
    target: list[torch.Tensor] | StateDict,
    target_type: str,
    reference_x: torch.Tensor,
) -> torch.Tensor | None:
    """Infer one iDLG pseudo-label and broadcast it across the attacked batch."""

    num_classes = _classification_num_classes(config, model=model, sample_x=reference_x[:1])
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    candidate_signals: list[tuple[str, torch.Tensor]] = []
    for name, _parameter in named_parameters:
        if name not in target:
            continue
        signal = _candidate_label_signal_from_tensor(target[name], num_classes)
        if signal is not None:
            candidate_signals.append((name, signal))
    if not candidate_signals:
        return None
    _preferred_name, preferred_signal = candidate_signals[-1]
    del target_type
    inferred = int(torch.argmax(preferred_signal).item())
    return torch.full((int(reference_x.shape[0]),), inferred, device=reference_x.device, dtype=torch.long)
