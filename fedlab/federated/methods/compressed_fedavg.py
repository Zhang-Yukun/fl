"""Compressed FedAvg sparse method implementation."""

from __future__ import annotations

from fedlab.federated.methods._sparse_common import SparseFedAvgMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.federated.protocol import resolve_upload_mode
from fedlab.utils.serialization import compress_topk


@federated_method('compressed_fedavg', compressed=True, description='Legacy top-k sparse FedAvg alias')
class CompressedFedAvgMethod(SparseFedAvgMethodBase):
    """Concrete legacy compressed FedAvg alias using Top-k sparse updates."""

    name = 'compressed_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=True, description='Legacy top-k sparse FedAvg alias')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'topk_fraction'}))

    def client_update(self, *, model, global_state, received_global_state=None, common: dict, evaluation_kwargs: dict, result_cls, client=None, **_: object):
        upload_mode = resolve_upload_mode((client.config if client is not None else {}))
        if upload_mode != 'update':
            raise ValueError('Sparse upload methods only support transport.upload_mode=update')
        base_state = received_global_state if received_global_state is not None else global_state
        update, buffer_update = self._split_updates(model=model, base_state=base_state)
        fraction = float((client.config if client is not None else {}).get('federated', {}).get('topk_fraction', 0.05))
        sparse = compress_topk(update, fraction)
        return self._build_sparse_result(
            sparse_update=sparse,
            buffer_update=buffer_update,
            common=common,
            evaluation_kwargs=evaluation_kwargs,
            result_cls=result_cls,
            aggregation_payload_kind='sparse_update',
            compressor='topk',
        )
