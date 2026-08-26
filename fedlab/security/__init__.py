"""Gradient reconstruction attacks and privacy evaluation helpers."""

from fedlab.security.registry import (
    attack_plugin,
    compute_recovery_metric_matrix,
    configured_attack_names,
    get_recovery_metric,
    list_registered_attacks,
    list_registered_recovery_metrics,
    normalize_recovery_metric_name,
    recovery_metric_plugin,
    register_attack,
    register_recovery_metric,
    resolve_recovery_objective,
    resolve_recovery_threshold,
    run_attacks,
)

__all__ = [
    'attack_plugin',
    'compute_recovery_metric_matrix',
    'configured_attack_names',
    'get_recovery_metric',
    'list_registered_attacks',
    'list_registered_recovery_metrics',
    'normalize_recovery_metric_name',
    'recovery_metric_plugin',
    'register_attack',
    'register_recovery_metric',
    'resolve_recovery_objective',
    'resolve_recovery_threshold',
    'run_attacks',
]
