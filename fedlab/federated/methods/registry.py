"""Registry for pluggable federated algorithm implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fedlab.federated.methods.base import FederatedMethod, MethodConfigSpec


@dataclass(frozen=True)
class RegisteredMethod:
    """Immutable metadata for one registered federated algorithm."""

    name: str
    factory: Callable[[], FederatedMethod]
    compressed: bool
    description: str
    config_spec: MethodConfigSpec


_METHOD_REGISTRY: dict[str, RegisteredMethod] = {}


def _resolve_config_spec(factory: Callable[[], FederatedMethod]) -> MethodConfigSpec:
    spec = getattr(factory, 'config_spec', MethodConfigSpec())
    if isinstance(spec, MethodConfigSpec):
        return spec
    return MethodConfigSpec(
        federated_keys=frozenset(getattr(spec, 'federated_keys', ())),
        root_blocks=frozenset(getattr(spec, 'root_blocks', ())),
        uses_privacy_block=bool(getattr(spec, 'uses_privacy_block', False)),
    )


def register_method(name: str, factory: Callable[[], FederatedMethod], *, compressed: bool = False, description: str = "") -> None:
    """Register one federated algorithm factory by name.

    Example:
        ``register_method("fedavg", FedAvgMethod, compressed=False)``.
    """

    key = str(name).lower()
    existing = _METHOD_REGISTRY.get(key)
    if existing is not None and existing.factory is not factory:
        raise ValueError(f"Federated method already registered: {key}")
    _METHOD_REGISTRY[key] = RegisteredMethod(
        name=key,
        factory=factory,
        compressed=bool(compressed),
        description=str(description),
        config_spec=_resolve_config_spec(factory),
    )


def federated_method(name: str, *, compressed: bool = False, description: str = ""):
    """Decorator form for ``register_method``."""

    def _decorator(factory: Callable[[], FederatedMethod]) -> Callable[[], FederatedMethod]:
        """Register the decorated federated method factory."""

        register_method(name, factory, compressed=compressed, description=description)
        return factory

    return _decorator


def get_registered_method(name: str) -> RegisteredMethod:
    """Return one registered federated algorithm descriptor."""

    key = str(name).lower()
    if key not in _METHOD_REGISTRY:
        raise ValueError(f"Unknown federated algorithm: {key}")
    return _METHOD_REGISTRY[key]


def build_method(name: str) -> FederatedMethod:
    """Instantiate one registered federated algorithm."""

    return get_registered_method(name).factory()


def is_registered_compressed(name: str) -> bool:
    """Return whether the named method is registered as compressed."""

    return get_registered_method(name).compressed


def list_registered_methods() -> tuple[RegisteredMethod, ...]:
    """Return all registered federated algorithms sorted by name."""

    return tuple(_METHOD_REGISTRY[name] for name in sorted(_METHOD_REGISTRY))
