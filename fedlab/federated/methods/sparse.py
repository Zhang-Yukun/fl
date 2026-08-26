"""Compatibility facade for sparse federated method registrations."""

from fedlab.federated.methods._sparse_common import SparseFedAvgMethodBase
from fedlab.federated.methods.compressed_fedavg import CompressedFedAvgMethod
from fedlab.federated.methods.dp_topk_fedavg import DpTopkFedAvgMethod
from fedlab.federated.methods.randomk_fedavg import RandomkFedAvgMethod
from fedlab.federated.methods.soteriafl import SoteriaFLMethod
from fedlab.federated.methods.sparse_fedavg import SparseFedAvgMethod

__all__ = [
    'SparseFedAvgMethodBase',
    'CompressedFedAvgMethod',
    'SparseFedAvgMethod',
    'DpTopkFedAvgMethod',
    'RandomkFedAvgMethod',
    'SoteriaFLMethod',
]
