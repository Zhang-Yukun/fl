"""Attack registry for pluggable reconstruction attack execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch


AttackFn = Callable[..., Any]
RecoveryMatrixFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
RecoveryThresholdFn = Callable[[dict[str, Any], float], float]


AttackValueGetter = Callable[[Any], float | None]


@dataclass(frozen=True)
class AttackSummaryMetricSpec:
    """One registered attack-summary metric definition."""

    name: str
    value_getter: AttackValueGetter
    average_key: str | None = None
    best_key: str | None = None
    best_objective: str = 'min'
    include_per_method: bool = True
    include_overall: bool = True
    privacy_direction: str | None = None


@dataclass(frozen=True)
class AttackRecordFieldSpec:
    """One registered JSON record field serializer."""

    output_key: str
    value_getter: Callable[[Any], Any]
    omit_if_none: bool = True
    finite_float_to_none: bool = True


@dataclass(frozen=True)
class AttackArtifactFieldSpec:
    """One registered artifact payload field serializer."""

    output_key: str
    value_getter: Callable[[Any], Any]
    tensor_to_cpu: bool = False


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
_ATTACK_SUMMARY_METRICS: dict[str, AttackSummaryMetricSpec] = {}
_BUILTIN_ATTACK_SUMMARY_METRICS_LOADED = False
_ATTACK_RECORD_FIELDS: dict[str, AttackRecordFieldSpec] = {}
_ATTACK_RECORD_FIELD_ORDER: list[str] = []
_BUILTIN_ATTACK_RECORD_FIELDS_LOADED = False
_ATTACK_ARTIFACT_FIELDS: dict[str, AttackArtifactFieldSpec] = {}
_ATTACK_ARTIFACT_FIELD_ORDER: list[str] = []
_BUILTIN_ATTACK_ARTIFACT_FIELDS_LOADED = False


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


def register_attack_summary_metric(
    name: str,
    value_getter: AttackValueGetter,
    *,
    average_key: str | None = None,
    best_key: str | None = None,
    best_objective: str = 'min',
    include_per_method: bool = True,
    include_overall: bool = True,
    privacy_direction: str | None = None,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> AttackSummaryMetricSpec:
    """Register one summary metric derived from ``AttackResult`` fields."""

    normalized_name = _normalize_name(name)
    if best_objective not in {'min', 'max'}:
        raise ValueError(f'Unsupported summary metric best_objective: {best_objective}')
    spec = AttackSummaryMetricSpec(
        name=normalized_name,
        value_getter=value_getter,
        average_key=average_key,
        best_key=best_key,
        best_objective=best_objective,
        include_per_method=bool(include_per_method),
        include_overall=bool(include_overall),
        privacy_direction=privacy_direction,
    )
    names = (normalized_name, *(_normalize_name(alias) for alias in aliases))
    for candidate in names:
        existing = _ATTACK_SUMMARY_METRICS.get(candidate)
        if existing is not None and existing != spec and not replace:
            raise ValueError(f'Attack summary metric alias already registered: {candidate}')
    for candidate in names:
        _ATTACK_SUMMARY_METRICS[candidate] = spec
    return spec


def attack_summary_metric_plugin(
    name: str,
    *,
    average_key: str | None = None,
    best_key: str | None = None,
    best_objective: str = 'min',
    include_per_method: bool = True,
    include_overall: bool = True,
    privacy_direction: str | None = None,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
):
    """Decorator for registering one attack summary metric."""

    def decorator(value_getter: AttackValueGetter) -> AttackValueGetter:
        register_attack_summary_metric(
            name,
            value_getter,
            average_key=average_key,
            best_key=best_key,
            best_objective=best_objective,
            include_per_method=include_per_method,
            include_overall=include_overall,
            privacy_direction=privacy_direction,
            aliases=aliases,
            replace=replace,
        )
        return value_getter

    return decorator


def _ensure_builtin_attack_summary_metrics_registered() -> None:
    """Register builtin attack summary metrics once."""

    global _BUILTIN_ATTACK_SUMMARY_METRICS_LOADED
    if _BUILTIN_ATTACK_SUMMARY_METRICS_LOADED:
        return
    _BUILTIN_ATTACK_SUMMARY_METRICS_LOADED = True
    register_attack_summary_metric(
        'reconstruction_mse',
        lambda result: getattr(result, 'mse', None) if getattr(result, 'metric_name', None) == 'reconstruction_mse' else getattr(result, 'exact_target_mse', None),
        average_key='overall_avg_exact_target_mse',
        privacy_direction='higher_is_more_private',
    )
    register_attack_summary_metric(
        'exact_target_mse',
        lambda result: getattr(result, 'exact_target_mse', None),
        average_key='overall_avg_exact_target_mse',
        privacy_direction=None,
    )
    register_attack_summary_metric(
        'nearest_client_train_mse',
        lambda result: getattr(result, 'nearest_client_train_mse', None),
        average_key='overall_avg_nearest_client_train_mse',
        privacy_direction='higher_is_more_private',
    )
    register_attack_summary_metric(
        'budget_recovered_fraction',
        lambda result: getattr(result, 'budget_recovered_fraction', None),
        average_key='overall_avg_budget_recovered_fraction',
        privacy_direction='lower_is_more_private',
        best_objective='max',
    )
    register_attack_summary_metric(
        'coverage_recovered_fraction',
        lambda result: getattr(result, 'coverage_recovered_fraction', None),
        average_key='overall_avg_coverage_recovered_fraction',
        privacy_direction='lower_is_more_private',
        best_objective='max',
    )
    register_attack_summary_metric(
        'psnr',
        lambda result: getattr(result, 'psnr', None),
        average_key='overall_avg_psnr',
        best_key='overall_best_psnr',
        best_objective='max',
        privacy_direction=None,
    )
    register_attack_summary_metric(
        'ssim',
        lambda result: getattr(result, 'ssim', None),
        average_key='overall_avg_ssim',
        best_key='overall_best_ssim',
        best_objective='max',
        privacy_direction=None,
    )
    register_attack_summary_metric(
        'objective_mse',
        lambda result: getattr(result, 'gradient_mse', None),
        average_key='overall_avg_objective_mse',
        privacy_direction=None,
    )
    register_attack_summary_metric(
        'time_seconds',
        lambda result: getattr(result, 'time_seconds', None),
        average_key='overall_avg_time_seconds',
        privacy_direction=None,
    )


def list_registered_attack_summary_metrics() -> dict[str, AttackSummaryMetricSpec]:
    """Return a snapshot of registered attack summary metrics."""

    _ensure_builtin_attack_summary_metrics_registered()
    return dict(_ATTACK_SUMMARY_METRICS)


def get_attack_summary_metric(name: str) -> AttackSummaryMetricSpec:
    """Return one registered attack summary metric spec."""

    _ensure_builtin_attack_summary_metrics_registered()
    normalized = _normalize_name(name)
    if normalized not in _ATTACK_SUMMARY_METRICS:
        raise ValueError(f'Unknown attack summary metric: {name}')
    return _ATTACK_SUMMARY_METRICS[normalized]


def attack_primary_metric_direction(name: str) -> str:
    """Return the privacy interpretation for one primary attack metric."""

    spec = get_attack_summary_metric(name)
    if spec.privacy_direction is not None:
        return spec.privacy_direction
    return 'higher_is_more_private'


def summarize_metric_values(results: list[Any], metric_name: str) -> dict[str, float | None]:
    """Compute average/best summary stats for one registered attack metric."""

    spec = get_attack_summary_metric(metric_name)
    values = [spec.value_getter(result) for result in results]
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    average = None if not finite else (sum(finite) / len(finite))
    if not finite:
        best = None
    elif spec.best_objective == 'max':
        best = max(finite)
    else:
        best = min(finite)
    return {'average': average, 'best': best}


def register_attack_record_field(
    output_key: str,
    value_getter: Callable[[Any], Any],
    *,
    omit_if_none: bool = True,
    finite_float_to_none: bool = True,
    replace: bool = False,
) -> AttackRecordFieldSpec:
    """Register one JSON attack record field serializer."""

    key = str(output_key)
    spec = AttackRecordFieldSpec(
        output_key=key,
        value_getter=value_getter,
        omit_if_none=bool(omit_if_none),
        finite_float_to_none=bool(finite_float_to_none),
    )
    existing = _ATTACK_RECORD_FIELDS.get(key)
    if existing is not None and existing != spec and not replace:
        raise ValueError(f'Attack record field already registered: {key}')
    _ATTACK_RECORD_FIELDS[key] = spec
    if key not in _ATTACK_RECORD_FIELD_ORDER:
        _ATTACK_RECORD_FIELD_ORDER.append(key)
    return spec


def register_attack_artifact_field(
    output_key: str,
    value_getter: Callable[[Any], Any],
    *,
    tensor_to_cpu: bool = False,
    replace: bool = False,
) -> AttackArtifactFieldSpec:
    """Register one attack artifact payload field serializer."""

    key = str(output_key)
    spec = AttackArtifactFieldSpec(
        output_key=key,
        value_getter=value_getter,
        tensor_to_cpu=bool(tensor_to_cpu),
    )
    existing = _ATTACK_ARTIFACT_FIELDS.get(key)
    if existing is not None and existing != spec and not replace:
        raise ValueError(f'Attack artifact field already registered: {key}')
    _ATTACK_ARTIFACT_FIELDS[key] = spec
    if key not in _ATTACK_ARTIFACT_FIELD_ORDER:
        _ATTACK_ARTIFACT_FIELD_ORDER.append(key)
    return spec


def _ensure_builtin_attack_record_fields_registered() -> None:
    global _BUILTIN_ATTACK_RECORD_FIELDS_LOADED
    if _BUILTIN_ATTACK_RECORD_FIELDS_LOADED:
        return
    _BUILTIN_ATTACK_RECORD_FIELDS_LOADED = True
    register_attack_record_field('name', lambda result: getattr(result, 'name', None))
    register_attack_record_field('psnr', lambda result: getattr(result, 'psnr', None), omit_if_none=False)
    register_attack_record_field('ssim', lambda result: getattr(result, 'ssim', None), omit_if_none=False)
    register_attack_record_field('iterations', lambda result: getattr(result, 'iterations', None))
    register_attack_record_field('time_seconds', lambda result: getattr(result, 'time_seconds', None))
    register_attack_record_field('success', lambda result: getattr(result, 'success', None))
    register_attack_record_field('success_threshold', lambda result: getattr(result, 'success_threshold', None))
    register_attack_record_field('target_type', lambda result: getattr(result, 'target_type', None))
    register_attack_record_field('exact_target_mse', lambda result: getattr(result, 'exact_target_mse', None))
    register_attack_record_field('nearest_client_train_mse', lambda result: getattr(result, 'nearest_client_train_mse', None))
    register_attack_record_field('nearest_client_train_indices', lambda result: getattr(result, 'nearest_client_train_indices', None))
    register_attack_record_field('matched_reference_indices', lambda result: getattr(result, 'matched_reference_indices', None))
    register_attack_record_field('matched_reference_metric_name', lambda result: getattr(result, 'matched_reference_metric_name', None))
    register_attack_record_field('matched_reference_metric_value', lambda result: getattr(result, 'matched_reference_metric_value', None))
    register_attack_record_field('recovered_count', lambda result: getattr(result, 'recovered_count', None))
    register_attack_record_field('reconstructed_count', lambda result: getattr(result, 'reconstructed_count', None))
    register_attack_record_field('reference_count', lambda result: getattr(result, 'reference_count', None))
    register_attack_record_field('budget_recovered_fraction', lambda result: getattr(result, 'budget_recovered_fraction', None))
    register_attack_record_field('coverage_recovered_fraction', lambda result: getattr(result, 'coverage_recovered_fraction', None))
    register_attack_record_field('client_id', lambda result: getattr(result, 'client_id', None))
    register_attack_record_field('round_index', lambda result: getattr(result, 'round_index', None))
    register_attack_record_field('sample_index', lambda result: getattr(result, 'sample_index', None))
    register_attack_record_field('artifact_path', lambda result: getattr(result, 'artifact_path', None))
    register_attack_record_field('primary_metric_name', lambda result: getattr(result, 'metric_name', None))
    register_attack_record_field('primary_metric_value', lambda result: getattr(result, 'mse', None), omit_if_none=False)
    register_attack_record_field('objective_mse', lambda result: getattr(result, 'gradient_mse', None), omit_if_none=False)


def _ensure_builtin_attack_artifact_fields_registered() -> None:
    global _BUILTIN_ATTACK_ARTIFACT_FIELDS_LOADED
    if _BUILTIN_ATTACK_ARTIFACT_FIELDS_LOADED:
        return
    _BUILTIN_ATTACK_ARTIFACT_FIELDS_LOADED = True
    for key in ('name', 'client_id', 'round_index', 'sample_index', 'target_type', 'reference_label'):
        register_attack_artifact_field(key, lambda result, attr=key: getattr(result, attr, None))
    register_attack_artifact_field('primary_metric_name', lambda result: getattr(result, 'metric_name', None))
    register_attack_artifact_field('primary_metric_value', lambda result: getattr(result, 'mse', None))
    for key in ('real_x', 'real_y', 'reference_x', 'reference_y', 'reconstructed_x', 'reconstructed_y'):
        register_attack_artifact_field(key, lambda result, attr=key: getattr(result, attr, None), tensor_to_cpu=True)
    for key in ('plot_real_x', 'plot_real_y', 'plot_reference_x', 'plot_reference_y', 'plot_reconstructed_x', 'plot_reconstructed_y'):
        register_attack_artifact_field(key, lambda result, attr=key: getattr(result, attr, None), tensor_to_cpu=True)
    for key in (
        'exact_target_mse',
        'nearest_client_train_mse',
        'nearest_client_train_indices',
        'matched_reference_indices',
        'matched_reference_metric_name',
        'matched_reference_metric_value',
        'recovered_count',
        'reconstructed_count',
        'reference_count',
        'budget_recovered_fraction',
        'coverage_recovered_fraction',
    ):
        register_attack_artifact_field(key, lambda result, attr=key: getattr(result, attr, None))


def list_registered_attack_record_fields() -> tuple[AttackRecordFieldSpec, ...]:
    _ensure_builtin_attack_record_fields_registered()
    return tuple(_ATTACK_RECORD_FIELDS[key] for key in _ATTACK_RECORD_FIELD_ORDER)


def list_registered_attack_artifact_fields() -> tuple[AttackArtifactFieldSpec, ...]:
    _ensure_builtin_attack_artifact_fields_registered()
    return tuple(_ATTACK_ARTIFACT_FIELDS[key] for key in _ATTACK_ARTIFACT_FIELD_ORDER)


def serialize_attack_record(result: Any) -> dict[str, Any]:
    """Serialize one attack result into a JSON-ready record via the registry."""

    record: dict[str, Any] = {}
    for spec in list_registered_attack_record_fields():
        value = spec.value_getter(result)
        if value is None and spec.omit_if_none:
            continue
        if isinstance(value, float) and spec.finite_float_to_none and not math.isfinite(value):
            value = None
        if value is None and spec.omit_if_none:
            continue
        record[spec.output_key] = value
    return record


def build_attack_artifact_payload(result: Any) -> dict[str, Any]:
    """Serialize one attack artifact payload via the registry."""

    payload: dict[str, Any] = {}
    for spec in list_registered_attack_artifact_fields():
        value = spec.value_getter(result)
        if spec.tensor_to_cpu and value is not None:
            value = value.detach().cpu()
        payload[spec.output_key] = value
    return payload


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
