"""Task-aware model builders."""

from __future__ import annotations

from typing import Any

from torch import nn


def _uniform_init_module(module: nn.Module, *, low: float, high: float) -> None:
    try:
        if hasattr(module, "weight") and getattr(module, "weight") is not None:
            module.weight.data.uniform_(low, high)
    except Exception:
        pass
    try:
        if hasattr(module, "bias") and getattr(module, "bias") is not None:
            module.bias.data.uniform_(low, high)
    except Exception:
        pass


def _apply_model_init(model: nn.Module, config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    init_name = str(model_cfg.get("init", "default")).strip().lower()
    if init_name in {"", "default", "none"}:
        return model
    if init_name in {"uniform", "public_dlg_uniform", "dlg_public_uniform"}:
        low = float(model_cfg.get("init_uniform_low", -0.5))
        high = float(model_cfg.get("init_uniform_high", 0.5))
        if not low < high:
            raise ValueError(f"Invalid uniform init range: low={low} high={high}")
        model.apply(lambda module: _uniform_init_module(module, low=low, high=high))
        return model
    raise ValueError(f"Unknown model init: {init_name}")


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model through the configured task plugin."""

    from fedlab.tasks.registry import annotate_model, get_task

    return annotate_model(_apply_model_init(get_task(config).build_model(config), config), config)


__all__ = ["build_model"]
