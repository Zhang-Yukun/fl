"""Compatibility facade for dense federated method registrations."""

from fedlab.federated.methods._dense_common import DenseFedAvgMethodBase
from fedlab.federated.methods.adaptive_clipped_rdp_fedavg import AdaptiveClippedRdpFedAvgMethod
from fedlab.federated.methods.fedavg import FedAvgMethod
from fedlab.federated.methods.fedaware import FedAwareMethod

__all__ = [
    'DenseFedAvgMethodBase',
    'FedAvgMethod',
    'FedAwareMethod',
    'AdaptiveClippedRdpFedAvgMethod',
]
