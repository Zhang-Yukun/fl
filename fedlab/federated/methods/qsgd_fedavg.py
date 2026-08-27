"""QSGD FedAvg implementation."""

from __future__ import annotations

from fedlab.federated.methods._quantized_common import QuantizedDenseMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.utils.serialization import add_update, average_states, dequantize_qsgd_state_update, quantize_qsgd_state_update, state_num_bytes, state_num_parameters, subtract_state


@federated_method('qsgd_fedavg', compressed=False, description='QSGD quantized dense upload FedAvg')
class QsgdFedAvgMethod(QuantizedDenseMethodBase):
    name = 'qsgd_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='QSGD quantized dense upload FedAvg')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'qsgd_levels', 'quantization_seed'}))

    def client_update(self, *, local_state, global_state, common: dict, evaluation_kwargs: dict, result_cls, client, round_index: int, **_: object):
        levels = int(client.config.get('federated', {}).get('qsgd_levels', 127))
        quantized = quantize_qsgd_state_update(
            subtract_state(local_state, global_state),
            levels=levels,
            generator=client._upload_quantization_generator(round_index),
        )
        return result_cls(
            **common,
            aggregation_state=quantized,
            **evaluation_kwargs,
            upload_bytes=state_num_bytes(quantized),
            upload_parameters=state_num_parameters(quantized),
            parameter_upload_bytes=state_num_bytes(quantized),
            parameter_upload_parameters=state_num_parameters(quantized),
            transport_upload_bytes=state_num_bytes(quantized),
            aggregation_payload_kind='qsgd_update',
            compressor=f'qsgd_{levels}_levels',
        )

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: object) -> list[float]:
        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        levels = int(server.config.get('federated', {}).get('qsgd_levels', 127))
        dense_updates = [dequantize_qsgd_state_update(result.aggregation_state, levels=levels) for result in results]
        averaged_update = average_states(dense_updates, sample_weights)
        server.global_state = add_update(server.global_state, averaged_update)
        return weights

    def extract_attack_payload(self, *, result, server=None, **kwargs: object):
        if result.aggregation_state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        levels = int(server.config.get('federated', {}).get('qsgd_levels', 127)) if server is not None else 127
        return dequantize_qsgd_state_update(result.aggregation_state, levels=levels)
