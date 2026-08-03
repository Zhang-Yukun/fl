"""Parameter-efficient tuning helpers for federated experiments.

Example:
    ``is_fedpetuning(config)`` toggles the PatchTST FedPETuning-style path,
    while ``serialize_trainable_state(model)`` exports only synchronized
    adapter/head parameters.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import torch

from federated_ts.utils.serialization import StateDict


def is_fedpetuning(config: dict) -> bool:
    """Return whether the config enables the FedPETuning-style path."""

    algorithm = str(config.get("federated", {}).get("algorithm", "fedavg")).lower()
    peft_cfg = config.get("model", {}).get("peft", {})
    return algorithm == "fedpetuning" or (
        bool(peft_cfg.get("enabled", False)) and str(peft_cfg.get("method", "")).lower() == "fedpetuning"
    )


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    """Return the ordered names of trainable model parameters."""

    return [name for name, param in model.named_parameters() if param.requires_grad]


def serialize_trainable_state(model: torch.nn.Module) -> StateDict:
    """Serialize only trainable parameters from a model.

    Example:
        ``serialize_trainable_state(model)`` produces the small PEFT payload
        synchronized by the FedPETuning-style algorithm.
    """

    trainable = set(trainable_parameter_names(model))
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
        if name in trainable
    )


def subset_state(state: StateDict, names: Iterable[str]) -> StateDict:
    """Select an ordered subset of a serialized state dict."""

    return OrderedDict((name, state[name].detach().cpu().clone()) for name in names)
