"""Pluggable federated algorithm interfaces and registry."""

from fedlab.federated.methods.base import FederatedMethod, MethodCapabilities
from fedlab.federated.methods.registry import build_method, get_registered_method, is_registered_compressed, list_registered_methods, register_method

# Import stub registrations eagerly so the registry reflects current algorithm coverage.
from fedlab.federated.methods import stubs as _stubs  # noqa: F401

__all__ = [
    'FederatedMethod',
    'MethodCapabilities',
    'build_method',
    'get_registered_method',
    'is_registered_compressed',
    'list_registered_methods',
    'register_method',
]
