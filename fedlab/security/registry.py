"""Attack registry for pluggable reconstruction attack execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch


AttackFn = Callable[..., Any]
RecoveryMatrixFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
RecoveryThresholdFn = Callable[[dict[str, Any], float], float]


@dataclass(frozen=True)
class RecoveryMetricSpec:
    """One registered set-recovery metric definition."""

    name: str
    compute_matrix: RecoveryMatrixFn
    default_objective: str
    default_threshold: RecoveryThresholdFn


_ATTACKS: dict[str, AttackFn] = {}
_ATTACK_ORDER: list[str] = []
_BUILTIN_ATTACKS_LOADED = False
_RECOVERY_METRICS: dict[str, RecoveryMetricSpec] = {}
_BUILTIN_RECOVERY_METRICS_LOADED = False


def _normalize_name(name: str) -> str:
    """Normalize registry names to canonical lowercase keys."""

    return str(name).strip().lower()


def register_attack(name: str, attack_fn: AttackFn, *aliases: str, replace: bool = False) -> AttackFn:
    """Register one attack function and optional aliases."""

    normalized_name = _normalize_name(name)
    names = (normalized_name, *(_normalize_name(alias) for alias in aliases))
    for candidate in names:
        existing = _ATTACKS.get(candidate)
        if existing is not None and existing is not attack_fn and not replace:
            raise ValueError(f"Attack alias already registered: {candidate}")
    for candidate in names:
        _ATTACKS[candidate] = attack_fn
    if normalized_name not in _ATTACK_ORDER:
        _ATTACK_ORDER.append(normalized_name)
    return attack_fn


def attack_plugin(name: str, *aliases: str, replace: bool = False):
    """Decorator for registering one attack function."""

    def decorator(attack_fn: AttackFn) -> AttackFn:
        return register_attack(name, attack_fn, *aliases, replace=replace)

    return decorator


def _ensure_builtin_attacks_registered() -> None:
    """Load builtin attacks lazily so monkeypatching ``attacks.py`` still works."""

    global _BUILTIN_ATTACKS_LOADED
    if _BUILTIN_ATTACKS_LOADED:
        return
    _BUILTIN_ATTACKS_LOADED = True

    def _dlg(*args, **kwargs):
        from fedlab.security import attacks as attacks_module

        return attacks_module.dlg_attack(*args, **kwargs)

    def _idlg(*args, **kwargs):
        from fedlab.security import attacks as attacks_module

        return attacks_module.idlg_attack(*args, **kwargs)

    register_attack('dlg', _dlg, 'deep_leakage_from_gradients')
    register_attack('idlg', _idlg, 'improved_deep_leakage_from_gradients')


def list_registered_attacks() -> dict[str, AttackFn]:
    """Return a snapshot of registered attacks."""

    _ensure_builtin_attacks_registered()
    return dict(_ATTACKS)


def configured_attack_names(config: dict[str, Any]) -> tuple[str, ...]:
    """Resolve attack method names from config, defaulting to DLG+iDLG."""

    _ensure_builtin_attacks_registered()
    attack_cfg = config.get('attack', {})
    configured = attack_cfg.get('methods')
    if configured is None:
        legacy = attack_cfg.get('method')
        if legacy is not None:
            configured = [legacy]
    if configured is None:
        return ('dlg', 'idlg')
    if isinstance(configured, str):
        value = configured.strip().lower()
        if value == 'all':
            return tuple(_ATTACK_ORDER)
        names = [item.strip().lower() for item in configured.split(',') if item.strip()]
    else:
        names = [str(item).strip().lower() for item in configured if str(item).strip()]
    if not names:
        raise ValueError('attack.methods must not be empty')
    unknown = [name for name in names if name not in _ATTACKS]
    if unknown:
        raise ValueError(f"Unknown attack methods: {unknown}")
    return tuple(names)


def run_attacks(
    config: dict[str, Any],
    model_state: Any,
    target: Any,
    real_x: Any,
    real_y: Any,
    device: Any,
    *,
    target_type: str | None = None,
    reference_inputs: Any = None,
    reference_targets: Any = None,
) -> list[Any]:
    """Execute the configured attacks in order and return their results."""

    _ensure_builtin_attacks_registered()
    results = []
    for name in configured_attack_names(config):
        attack_fn = _ATTACKS[name]
        results.append(
            attack_fn(
                config,
                model_state,
                target,
                real_x,
                real_y,
                device,
                target_type=target_type,
                reference_inputs=reference_inputs,
                reference_targets=reference_targets,
            )
        )
    return results


def _attack_threshold(config: dict[str, Any]) -> float:
    return float(config.get('attack', {}).get('success_mse_threshold', 0.5))


def _attack_ssim_threshold(config: dict[str, Any]) -> float | None:
    value = config.get('attack', {}).get('success_ssim_threshold')
    if value is None:
        return None
    return float(value)


def _pairwise_mse_matrix(reconstructed: torch.Tensor, reference_inputs: torch.Tensor, _data_range: float) -> torch.Tensor:
    recon = reconstructed.detach().cpu().float().reshape(reconstructed.shape[0], -1)
    refs = reference_inputs.detach().cpu().float().reshape(reference_inputs.shape[0], -1)
    return torch.mean((recon[:, None, :] - refs[None, :, :]) ** 2, dim=-1)


def _compute_psnr_from_mse(mse_value: float, data_range: float) -> float:
    if mse_value <= 0:
        return float('inf')
    return float(20.0 * math.log10(data_range / math.sqrt(mse_value)))


def _pairwise_psnr_matrix(reconstructed: torch.Tensor, reference_inputs: torch.Tensor, data_range: float) -> torch.Tensor:
    mse_matrix = _pairwise_mse_matrix(reconstructed, reference_inputs, data_range)
    safe = torch.clamp(mse_matrix, min=torch.finfo(mse_matrix.dtype).tiny)
    values = 20.0 * math.log10(data_range) - 10.0 * torch.log10(safe)
    return torch.where(mse_matrix <= 0, torch.full_like(values, float('inf')), values)


def _pairwise_ssim_matrix(reconstructed: torch.Tensor, reference_inputs: torch.Tensor, data_range: float) -> torch.Tensor:
    from fedlab.security import attacks as attacks_module

    recon = reconstructed.detach().cpu().float()
    refs = reference_inputs.detach().cpu().float()
    values = torch.empty((recon.shape[0], refs.shape[0]), dtype=recon.dtype)
    for row in range(recon.shape[0]):
        for col in range(refs.shape[0]):
            values[row, col] = attacks_module._compute_ssim(recon[row : row + 1], refs[col : col + 1], data_range)
    return values


def register_recovery_metric(
    name: str,
    compute_matrix: RecoveryMatrixFn,
    *,
    default_objective: str,
    default_threshold: RecoveryThresholdFn,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> RecoveryMetricSpec:
    """Register one set-recovery evaluation metric and optional aliases."""

    normalized_name = _normalize_name(name)
    if default_objective not in {'min', 'max'}:
        raise ValueError(f'Unsupported recovery metric objective: {default_objective}')
    spec = RecoveryMetricSpec(
        name=normalized_name,
        compute_matrix=compute_matrix,
        default_objective=default_objective,
        default_threshold=default_threshold,
    )
    names = (normalized_name, *(_normalize_name(alias) for alias in aliases))
    for candidate in names:
        existing = _RECOVERY_METRICS.get(candidate)
        if existing is not None and existing != spec and not replace:
            raise ValueError(f'Recovery metric alias already registered: {candidate}')
    for candidate in names:
        _RECOVERY_METRICS[candidate] = spec
    return spec


def recovery_metric_plugin(
    name: str,
    *,
    default_objective: str,
    default_threshold: RecoveryThresholdFn,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
):
    """Decorator for registering one recovery metric."""

    def decorator(compute_matrix: RecoveryMatrixFn) -> RecoveryMatrixFn:
        register_recovery_metric(
            name,
            compute_matrix,
            default_objective=default_objective,
            default_threshold=default_threshold,
            aliases=aliases,
            replace=replace,
        )
        return compute_matrix

    return decorator


def _ensure_builtin_recovery_metrics_registered() -> None:
    """Register builtin mse/psnr/ssim recovery metrics once."""

    global _BUILTIN_RECOVERY_METRICS_LOADED
    if _BUILTIN_RECOVERY_METRICS_LOADED:
        return
    _BUILTIN_RECOVERY_METRICS_LOADED = True
    register_recovery_metric('mse', _pairwise_mse_matrix, default_objective='min', default_threshold=lambda config, _data_range: _attack_threshold(config))
    register_recovery_metric('psnr', _pairwise_psnr_matrix, default_objective='max', default_threshold=lambda config, data_range: _compute_psnr_from_mse(_attack_threshold(config), data_range))
    register_recovery_metric('ssim', _pairwise_ssim_matrix, default_objective='max', default_threshold=lambda config, _data_range: 0.0 if _attack_ssim_threshold(config) is None else float(_attack_ssim_threshold(config)))


def list_registered_recovery_metrics() -> dict[str, RecoveryMetricSpec]:
    """Return a snapshot of registered recovery metrics."""

    _ensure_builtin_recovery_metrics_registered()
    return dict(_RECOVERY_METRICS)


def get_recovery_metric(name: str) -> RecoveryMetricSpec:
    """Return one registered recovery metric spec."""

    _ensure_builtin_recovery_metrics_registered()
    normalized = _normalize_name(name)
    if normalized not in _RECOVERY_METRICS:
        raise ValueError(f'Unsupported attack recovery metric: {name}')
    return _RECOVERY_METRICS[normalized]


def normalize_recovery_metric_name(value: Any, default: str = 'mse') -> str:
    """Resolve one configured recovery metric name."""

    metric = _normalize_name(default if value is None else value)
    if metric == 'auto':
        metric = _normalize_name(default)
    return get_recovery_metric(metric).name


def resolve_recovery_objective(value: Any, metric: str) -> str:
    """Resolve matching/success direction for one recovery metric."""

    objective = _normalize_name('auto' if value is None else value)
    if objective == 'auto':
        return get_recovery_metric(metric).default_objective
    if objective not in {'min', 'max'}:
        raise ValueError(f'Unsupported attack recovery objective: {objective}')
    return objective


def resolve_recovery_threshold(config: dict[str, Any], metric: str, data_range: float) -> float:
    """Resolve the configured recovery success threshold."""

    configured = config.get('attack', {}).get('recovery_success_threshold')
    if configured is not None:
        return float(configured)
    return float(get_recovery_metric(metric).default_threshold(config, data_range))


def compute_recovery_metric_matrix(reconstructed: torch.Tensor, reference_inputs: torch.Tensor, metric: str, data_range: float) -> torch.Tensor:
    """Return the pairwise matrix for one registered recovery metric."""

    return get_recovery_metric(metric).compute_matrix(reconstructed, reference_inputs, data_range)
