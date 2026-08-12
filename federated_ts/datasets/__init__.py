"""Task-aware dataset loaders."""

from __future__ import annotations

from typing import Any


def build_federated_loaders(config: dict[str, Any]):
    """Build per-client train loaders plus shared validation/test loaders."""

    from federated_ts.tasks.registry import get_task

    return get_task(config).build_federated_loaders(config)


__all__ = ["build_federated_loaders"]
