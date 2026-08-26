"""Pluggable federated algorithm interfaces and registry."""

from fedlab.federated.methods.base import FederatedMethod, MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import build_method, get_registered_method, is_registered_compressed, list_registered_methods, register_method

# Import migrated methods and then stub registrations to keep the registry complete.
from fedlab.federated.methods import dense as _dense  # noqa: F401
from fedlab.federated.methods import quantized as _quantized  # noqa: F401
from fedlab.federated.methods import sparse as _sparse  # noqa: F401
from fedlab.federated.methods import encoded as _encoded  # noqa: F401

__all__ = [
    'FederatedMethod',
    'MethodCapabilities',
    'MethodConfigSpec',
    'build_method',
    'get_registered_method',
    'is_registered_compressed',
    'list_registered_methods',
    'register_method',
]
