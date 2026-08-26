"""FedAware dense method implementation."""

from __future__ import annotations

from collections import OrderedDict

from fedlab.federated.methods._dense_common import DenseFedAvgMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.federated.protocol import derive_update_from_upload_payload, weighted_protocol_base_state
from fedlab.utils.aggregation import fedaware_weights
from fedlab.utils.serialization import add_update


@federated_method('fedaware', compressed=False, description='Adaptive weighted dense FedAvg aggregation')
class FedAwareMethod(DenseFedAvgMethodBase):
    """Concrete FedAware implementation on the method API."""

    name = 'fedaware'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Adaptive weighted dense FedAvg aggregation')
    config_spec = MethodConfigSpec(root_blocks=frozenset({'fedaware'}))

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: object) -> list[float]:
        """Aggregate dense updates with FedAware's learned aggregation weights."""

        aware_cfg = server.config.get('fedaware', {})
        sample_weights = [result.num_samples for result in results]
        updates = [
            derive_update_from_upload_payload(
                result.aggregation_state,
                server.method.reconstruct_received_global_state(
                    server=server,
                    global_state=round_base_state,
                    client_id=result.client_id,
                    round_index=round_index,
                    round_context=round_context or {},
                ),
                result.upload_mode,
            )
            for result in results
        ]
        weights = fedaware_weights(
            updates,
            sample_weights,
            alpha=float(aware_cfg.get('alpha', 0.5)),
            steps=int(aware_cfg.get('steps', 50)),
            lr=float(aware_cfg.get('lr', 0.1)),
        )
        averaged_update = None
        for update, weight in zip(updates, weights):
            scaled = OrderedDict((name, tensor * weight) for name, tensor in update.items())
            averaged_update = scaled if averaged_update is None else OrderedDict(
                (name, averaged_update[name] + scaled[name]) for name in scaled
            )
        protocol_base_state = weighted_protocol_base_state(server, results, round_base_state, round_index, round_context or {})
        server.global_state = add_update(protocol_base_state, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        return weights
