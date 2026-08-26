"""DP Top-k FedAvg method implementation."""

from __future__ import annotations

from fedlab.federated.methods._sparse_common import SparseFedAvgMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.federated.protocol import resolve_upload_mode
from fedlab.utils.serialization import compress_topk, privatize_state_update


@federated_method('dp_topk_fedavg', compressed=True, description='Top-k sparse FedAvg with DP preprocessing')
class DpTopkFedAvgMethod(SparseFedAvgMethodBase):
    name = 'dp_topk_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=True, description='Top-k sparse FedAvg with DP preprocessing')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'topk_fraction'}), uses_privacy_block=True)

    def client_update(self, *, model, global_state, received_global_state=None, common: dict, evaluation_kwargs: dict, result_cls, client, **_: object):
        upload_mode = resolve_upload_mode(client.config)
        if upload_mode != 'update':
            raise ValueError('Sparse upload methods only support transport.upload_mode=update')
        base_state = received_global_state if received_global_state is not None else global_state
        update, buffer_update = self._split_updates(model=model, base_state=base_state)
        fraction = float(client.config.get('federated', {}).get('topk_fraction', 0.05))
        privacy_cfg = client.config.get('privacy', {})
        privacy_clip_norm = float(privacy_cfg.get('clip_norm', 1.0))
        privacy_noise_multiplier = float(privacy_cfg.get('noise_multiplier', 0.1))
        update = privatize_state_update(update, privacy_clip_norm, privacy_noise_multiplier)
        sparse = compress_topk(update, fraction)
        return self._build_sparse_result(
            sparse_update=sparse,
            buffer_update=buffer_update,
            common=common,
            evaluation_kwargs=evaluation_kwargs,
            result_cls=result_cls,
            aggregation_payload_kind='dp_topk_dp_update',
            compressor='topk_dp',
            privacy_clip_norm=privacy_clip_norm,
            privacy_noise_multiplier=privacy_noise_multiplier,
        )
