"""Attack registry for pluggable reconstruction attack execution."""

from __future__ import annotations

from typing import Any, Callable


AttackFn = Callable[..., Any]

_ATTACKS: dict[str, AttackFn] = {}
_ATTACK_ORDER: list[str] = []
_BUILTIN_ATTACKS_LOADED = False


def _normalize_name(name: str) -> str:
    """Normalize attack names to a canonical lowercase key."""

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
