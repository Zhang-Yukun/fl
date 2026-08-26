"""Plotting policy registry for saved attack reconstruction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AttackPlotPolicy:
    """Visualization behavior for one attack name."""

    name: str
    show_real_y: bool = True
    show_reconstructed_y: bool = True


_PLOT_POLICIES: dict[str, AttackPlotPolicy] = {}
_BUILTIN_PLOT_POLICIES_LOADED = False


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def register_attack_plot_policy(
    name: str,
    *,
    show_real_y: bool = True,
    show_reconstructed_y: bool = True,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
) -> AttackPlotPolicy:
    """Register one plotting policy for one attack name."""

    policy = AttackPlotPolicy(
        name=_normalize_name(name),
        show_real_y=bool(show_real_y),
        show_reconstructed_y=bool(show_reconstructed_y),
    )
    names = (policy.name, *(_normalize_name(alias) for alias in aliases))
    for candidate in names:
        existing = _PLOT_POLICIES.get(candidate)
        if existing is not None and existing != policy and not replace:
            raise ValueError(f'Attack plot policy already registered: {candidate}')
    for candidate in names:
        _PLOT_POLICIES[candidate] = policy
    return policy


def attack_plot_policy_plugin(
    name: str,
    *,
    show_real_y: bool = True,
    show_reconstructed_y: bool = True,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
):
    """Decorator that registers one attack plotting policy."""

    def decorator(factory: Callable[[], Any]) -> Callable[[], Any]:
        del factory
        register_attack_plot_policy(
            name,
            show_real_y=show_real_y,
            show_reconstructed_y=show_reconstructed_y,
            aliases=aliases,
            replace=replace,
        )
        return factory

    return decorator


def _ensure_builtin_plot_policies_registered() -> None:
    global _BUILTIN_PLOT_POLICIES_LOADED
    if _BUILTIN_PLOT_POLICIES_LOADED:
        return
    _BUILTIN_PLOT_POLICIES_LOADED = True
    register_attack_plot_policy('dlg', aliases=('DLG',))
    register_attack_plot_policy('idlg', show_real_y=False, show_reconstructed_y=False, aliases=('iDLG',))


def list_registered_attack_plot_policies() -> dict[str, AttackPlotPolicy]:
    _ensure_builtin_plot_policies_registered()
    return dict(_PLOT_POLICIES)


def get_attack_plot_policy(name: str | None) -> AttackPlotPolicy:
    _ensure_builtin_plot_policies_registered()
    normalized = _normalize_name('attack' if name is None else name)
    return _PLOT_POLICIES.get(normalized, AttackPlotPolicy(name=normalized))


def should_plot_real_y(record: dict[str, Any], *, show_policy_overrides: bool = False) -> bool:
    if show_policy_overrides:
        return True
    return get_attack_plot_policy(str(record.get('name', 'attack'))).show_real_y


def should_plot_reconstructed_y(record: dict[str, Any], *, show_policy_overrides: bool = False) -> bool:
    if show_policy_overrides:
        return True
    return get_attack_plot_policy(str(record.get('name', 'attack'))).show_reconstructed_y


def discover_attack_names(records: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return attack names in stable first-seen order from saved records."""

    seen: list[str] = []
    for record in records:
        name = str(record.get('name', '')).strip()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)
