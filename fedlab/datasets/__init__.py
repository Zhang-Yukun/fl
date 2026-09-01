"""Task-aware dataset loaders."""

from __future__ import annotations

from typing import Any


def build_federated_loaders(config: dict[str, Any]):
    """Build per-client train loaders plus shared validation/test loaders."""

    from fedlab.tasks.registry import get_task

    return get_task(config).build_federated_loaders(config)


def build_server_evaluation_loaders(
    config: dict[str, Any],
    registration_metadata: dict[str, dict[str, Any]] | None = None,
):
    """Build only the server-side validation/test loaders required for FL evaluation."""

    from fedlab.tasks.registry import task_type

    resolved_task = task_type(config)
    if resolved_task == 'classification':
        from fedlab.datasets.image_classification import build_server_image_classification_evaluation_loaders

        return build_server_image_classification_evaluation_loaders(config)
    if resolved_task == 'forecasting':
        from fedlab.datasets.rare_earth import build_server_rare_earth_evaluation_loaders

        return build_server_rare_earth_evaluation_loaders(config, registration_metadata=registration_metadata)
    _, val_loader, test_loader = build_federated_loaders(config)
    return val_loader, test_loader


__all__ = ['build_federated_loaders', 'build_server_evaluation_loaders']
