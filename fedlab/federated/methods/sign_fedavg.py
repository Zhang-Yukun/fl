"""Sign FedAvg implementation."""

from __future__ import annotations

from fedlab.federated.methods._quantized_common import QuantizedDenseMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.utils.serialization import add_update, average_states, dequantize_state_update, quantize_state_update, state_num_bytes, state_num_parameters, subtract_state


@federated_method('sign_fedavg', compressed=False, description='Sign-based dense upload FedAvg')
class SignFedAvgMethod(QuantizedDenseMethodBase):
    name = 'sign_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Sign-based dense upload FedAvg')
    config_spec = MethodConfigSpec()

    def client_update(self, *, local_state, global_state, common: dict, evaluation_kwargs: dict, result_cls, **_: object):
        quantized = quantize_state_update(subtract_state(local_state, global_state), dtype='sign')
        return result_cls(
            **common,
            aggregation_state=quantized,
            **evaluation_kwargs,
            upload_bytes=state_num_bytes(quantized),
            upload_parameters=state_num_parameters(quantized),
            parameter_upload_bytes=state_num_bytes(quantized),
            parameter_upload_parameters=state_num_parameters(quantized),
            transport_upload_bytes=state_num_bytes(quantized),
            aggregation_payload_kind='sign_update',
            compressor='sign_mean_abs',
        )

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: object) -> list[float]:
        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        dense_updates = [dequantize_state_update(result.aggregation_state) for result in results]
        averaged_update = average_states(dense_updates, sample_weights)
        server.global_state = add_update(server.global_state, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        return weights

    def extract_attack_payload(self, *, result, server=None, **kwargs: object):
        if result.aggregation_state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        return dequantize_state_update(result.aggregation_state)
