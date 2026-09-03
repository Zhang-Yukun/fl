"""Classification-specific helpers for reconstruction attacks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fedlab.utils.serialization import StateDict


def is_classification_attack(config: dict[str, Any], target_template: torch.Tensor | None = None) -> bool:
    """Return whether the current attack target is a classification batch."""

    task_type = str(config.get("task", {}).get("type", "")).lower()
    if task_type == "classification":
        return True
    if target_template is None:
        return False
    return target_template.ndim == 1 and not torch.is_floating_point(target_template)


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


def _last_linear_parameter_names(model: nn.Module) -> tuple[str, str | None] | None:
    """Return parameter names for the final linear classifier head."""

    last_module_name = None
    last_module = None
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_module_name = module_name
            last_module = module
    if last_module_name is None or last_module is None:
        return None
    bias_name = f"{last_module_name}.bias" if last_module.bias is not None else None
    return f"{last_module_name}.weight", bias_name


def _target_type_prefers_argmin(target_type: str) -> bool:
    normalized = str(target_type).lower()
    return normalized in {"gradient", "gradients", "shared_gradient", "shared_gradients"}


def _preferred_label_signal(
    model: nn.Module,
    target: list[torch.Tensor] | StateDict,
    num_classes: int,
) -> torch.Tensor | None:
    last_linear_names = _last_linear_parameter_names(model)
    if last_linear_names is None:
        return None
    weight_name, bias_name = last_linear_names
    if bias_name is not None and bias_name in target:
        bias_signal = _candidate_label_signal_from_tensor(target[bias_name], num_classes)
        if bias_signal is not None:
            return bias_signal
    if weight_name in target:
        weight_signal = _candidate_label_signal_from_tensor(target[weight_name], num_classes)
        if weight_signal is not None:
            return weight_signal
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
    preferred_signal = _preferred_label_signal(model, target, num_classes)
    if preferred_signal is None:
        return None
    if _target_type_prefers_argmin(target_type):
        inferred = int(torch.argmin(preferred_signal).item())
    else:
        inferred = int(torch.argmax(preferred_signal).item())
    return torch.full((int(reference_x.shape[0]),), inferred, device=reference_x.device, dtype=torch.long)
