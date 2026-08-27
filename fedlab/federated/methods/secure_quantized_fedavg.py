"""Secure quantized FedAvg implementation."""

from __future__ import annotations

from fedlab.federated.methods._quantized_common import QuantizedDenseMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.federated.protocol import weighted_protocol_base_state
from fedlab.utils.serialization import add_update, average_states, dequantize_state_update, privatize_state_update, quantize_state_update, state_num_bytes, state_num_parameters, subtract_state


@federated_method('secure_quantized_fedavg', compressed=False, description='Dense quantized upload FedAvg')
class SecureQuantizedFedAvgMethod(QuantizedDenseMethodBase):
    name = 'secure_quantized_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Dense quantized upload FedAvg')
    config_spec = MethodConfigSpec(
        federated_keys=frozenset({'quantization_dtype', 'quantization_stochastic_rounding', 'quantization_seed'}),
        uses_privacy_block=True,
    )

    def uses_custom_download_transport(self) -> bool:
        return True

    def prepare_client_state(self, *, global_state, client, round_index: int, round_context: dict[str, object]):
        del round_index, round_context
        quantization_dtype = str(client.config.get('federated', {}).get('quantization_dtype', 'float16'))
        download_state = quantize_state_update(global_state, dtype=quantization_dtype)
        received_state = dequantize_state_update(download_state)
        return download_state, received_state

    def reconstruct_received_global_state(self, *, server, global_state, client_id: str, round_index: int, round_context: dict[str, object]):
        del client_id, round_index, round_context
        quantization_dtype = str(server.config.get('federated', {}).get('quantization_dtype', 'float16'))
        return dequantize_state_update(quantize_state_update(global_state, dtype=quantization_dtype))

    def client_update(self, *, local_state, received_global_state, common: dict, evaluation_kwargs: dict, result_cls, client, round_index: int, **_: object):
        update = subtract_state(local_state, received_global_state)
        privacy_cfg = client.config.get('privacy', {})
        privacy_clip_norm = float(privacy_cfg.get('clip_norm', 0.0))
        privacy_noise_multiplier = float(privacy_cfg.get('noise_multiplier', 0.0))
        update = privatize_state_update(update, privacy_clip_norm, privacy_noise_multiplier)
        quantized = quantize_state_update(
            update,
            dtype=str(client.config.get('federated', {}).get('quantization_dtype', 'float16')),
            stochastic_rounding=bool(client.config.get('federated', {}).get('quantization_stochastic_rounding', False)),
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
            aggregation_payload_kind='quantized_update',
            compressor=str(client.config.get('federated', {}).get('quantization_dtype', 'float16')) + '_quantized_dense',
            privacy_clip_norm=privacy_clip_norm,
            privacy_noise_multiplier=privacy_noise_multiplier,
        )

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: object) -> list[float]:
        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        dense_updates = [dequantize_state_update(result.aggregation_state) for result in results]
        averaged_update = average_states(dense_updates, sample_weights)
        protocol_base_state = weighted_protocol_base_state(server, results, round_base_state, round_index, round_context or {})
        server.global_state = add_update(protocol_base_state, averaged_update)
        return weights

    def extract_attack_payload(self, *, result, **kwargs: object):
        if result.aggregation_state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        return dequantize_state_update(result.aggregation_state)
