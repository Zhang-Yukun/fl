"""Compatibility facade for dense federated method registrations."""

from fedlab.federated.methods._dense_common import DenseFedAvgMethodBase
from fedlab.federated.methods.adaptive_clipped_rdp_fedavg import AdaptiveClippedRdpFedAvgMethod
from fedlab.federated.methods.fedavg import FedAvgMethod

__all__ = [
    'DenseFedAvgMethodBase',
    'FedAvgMethod',
    'AdaptiveClippedRdpFedAvgMethod',
]
