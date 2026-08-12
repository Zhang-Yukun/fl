"""Task-aware model builders."""

from __future__ import annotations

from typing import Any

from torch import nn


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model through the configured task plugin."""

    from fedlab.tasks.registry import annotate_model, get_task

    return annotate_model(get_task(config).build_model(config), config)


__all__ = ["build_model"]
