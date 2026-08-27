"""Random-k FedAvg method implementation."""

from __future__ import annotations

from fedlab.federated.methods._sparse_common import SparseFedAvgMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.utils.serialization import compress_randomk


@federated_method('randomk_fedavg', compressed=True, description='Random-k sparse FedAvg baseline')
class RandomkFedAvgMethod(SparseFedAvgMethodBase):
    name = 'randomk_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=True, description='Random-k sparse FedAvg baseline')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'topk_fraction', 'randomk_seed'}))

    def client_update(self, *, model, global_state, received_global_state=None, common: dict, evaluation_kwargs: dict, result_cls, client, round_index: int, **_: object):
        base_state = received_global_state if received_global_state is not None else global_state
        update, buffer_update = self._split_updates(model=model, base_state=base_state)
        fraction = float(client.config.get('federated', {}).get('topk_fraction', 0.05))
        sparse = compress_randomk(update, fraction, generator=client._randomk_generator(round_index))
        return self._build_sparse_result(
            sparse_update=sparse,
            buffer_update=buffer_update,
            common=common,
            evaluation_kwargs=evaluation_kwargs,
            result_cls=result_cls,
            aggregation_payload_kind='randomk_update',
            compressor='randomk_unbiased',
        )
