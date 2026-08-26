"""FedAvg dense method implementation."""

from __future__ import annotations

from fedlab.federated.methods._dense_common import DenseFedAvgMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.federated.protocol import derive_update_from_upload_payload, weighted_protocol_base_state
from fedlab.utils.serialization import add_update, average_states


@federated_method('fedavg', compressed=False, description='Standard dense FedAvg baseline')
class FedAvgMethod(DenseFedAvgMethodBase):
    """Concrete dense FedAvg implementation on the method API."""

    name = 'fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Standard dense FedAvg baseline')
    config_spec = MethodConfigSpec()

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: object) -> list[float]:
        """Aggregate dense client payloads with sample-size-weighted FedAvg."""

        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
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
        averaged_update = average_states(updates, sample_weights)
        protocol_base_state = weighted_protocol_base_state(server, results, round_base_state, round_index, round_context or {})
        server.global_state = add_update(protocol_base_state, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        return weights
