"""Compatibility facade for quantized federated method registrations."""

from fedlab.federated.methods._quantized_common import QuantizedDenseMethodBase
from fedlab.federated.methods.qsgd_fedavg import QsgdFedAvgMethod
from fedlab.federated.methods.secure_quantized_fedavg import SecureQuantizedFedAvgMethod
from fedlab.federated.methods.sign_fedavg import SignFedAvgMethod

__all__ = [
    'QuantizedDenseMethodBase',
    'SecureQuantizedFedAvgMethod',
    'SignFedAvgMethod',
    'QsgdFedAvgMethod',
]
