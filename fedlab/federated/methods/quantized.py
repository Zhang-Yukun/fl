"""Quantized dense federated algorithm implementations."""

from __future__ import annotations

from typing import Any

from fedlab.federated.methods.base import FederatedMethod, MethodCapabilities
from fedlab.federated.methods.registry import federated_method
from fedlab.utils.serialization import (
    add_update,
    average_states,
    dequantize_qsgd_state_update,
    dequantize_state_update,
    privatize_state_update,
    quantize_qsgd_state_update,
    quantize_state_update,
    subtract_state,
    state_num_bytes,
    state_num_parameters,
)


class _QuantizedDenseMethod(FederatedMethod):
    """Shared helpers for dense-upload quantized FedAvg variants."""

    def _dense_attack_payload(self, *, result, **_: Any):
        """Return the uploaded dense payload view exposed to the server."""

        if result.state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        return result.state


@federated_method('secure_quantized_fedavg', compressed=False, description='Dense quantized upload FedAvg')
class SecureQuantizedFedAvgMethod(_QuantizedDenseMethod):
    """Dense quantized FedAvg implementation on the method API."""

    name = 'secure_quantized_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Dense quantized upload FedAvg')

    def prepare_client_state(self, *, global_state, client, round_index: int, round_context: dict[str, Any]):
        """Quantize the download payload before the client loads it for training."""

        quantization_dtype = str(client.config.get('federated', {}).get('quantization_dtype', 'float16'))
        download_state = quantize_state_update(global_state, dtype=quantization_dtype)
        return download_state, dequantize_state_update(download_state)

    def client_update(
        self,
        *,
        local_state,
        received_global_state,
        common: dict[str, Any],
        evaluation_kwargs: dict[str, Any],
        result_cls,
        client,
        round_index: int,
        **_: Any,
    ):
        """Return a quantized dense update payload from one local model state."""

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
            state=quantized,
            **evaluation_kwargs,
            upload_bytes=state_num_bytes(quantized),
            upload_parameters=state_num_parameters(quantized),
            parameter_upload_bytes=state_num_bytes(quantized),
            parameter_upload_parameters=state_num_parameters(quantized),
            transport_upload_bytes=state_num_bytes(quantized),
            payload_kind='quantized_update',
            compressor=str(client.config.get('federated', {}).get('quantization_dtype', 'float16')) + '_quantized_dense',
            privacy_clip_norm=privacy_clip_norm,
            privacy_noise_multiplier=privacy_noise_multiplier,
        )

    def aggregate(self, *, server, results, round_base_state=None, **_: Any) -> list[float]:
        """Aggregate dequantized client updates and reapply the download quantization view."""

        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        dense_updates = [dequantize_state_update(result.state) for result in results]
        averaged_update = average_states(dense_updates, sample_weights)
        quantization_dtype = str(server.config.get('federated', {}).get('quantization_dtype', 'float16'))
        compressed_base = dequantize_state_update(quantize_state_update(server.global_state, dtype=quantization_dtype))
        server.global_state = add_update(compressed_base, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        return weights

    def extract_attack_payload(self, *, result, **kwargs: Any):
        """Expose the dequantized dense payload to the server attacker."""

        if result.state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        return dequantize_state_update(result.state)


@federated_method('sign_fedavg', compressed=False, description='Sign-based dense upload FedAvg')
class SignFedAvgMethod(_QuantizedDenseMethod):
    """Sign-based dense FedAvg implementation on the method API."""

    name = 'sign_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Sign-based dense upload FedAvg')

    def client_update(self, *, local_state, global_state, common: dict[str, Any], evaluation_kwargs: dict[str, Any], result_cls, **_: Any):
        """Return a sign-compressed dense update payload from one local model state."""

        quantized = quantize_state_update(subtract_state(local_state, global_state), dtype='sign')
        return result_cls(
            **common,
            state=quantized,
            **evaluation_kwargs,
            upload_bytes=state_num_bytes(quantized),
            upload_parameters=state_num_parameters(quantized),
            parameter_upload_bytes=state_num_bytes(quantized),
            parameter_upload_parameters=state_num_parameters(quantized),
            transport_upload_bytes=state_num_bytes(quantized),
            payload_kind='sign_update',
            compressor='sign_mean_abs',
        )

    def aggregate(self, *, server, results, round_base_state=None, **_: Any) -> list[float]:
        """Aggregate dequantized sign uploads with standard FedAvg weighting."""

        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        dense_updates = [dequantize_state_update(result.state) for result in results]
        averaged_update = average_states(dense_updates, sample_weights)
        server.global_state = add_update(server.global_state, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        return weights

    def extract_attack_payload(self, *, result, **kwargs: Any):
        """Expose the dequantized sign payload to the server attacker."""

        if result.state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        return dequantize_state_update(result.state)


@federated_method('qsgd_fedavg', compressed=False, description='QSGD quantized dense upload FedAvg')
class QsgdFedAvgMethod(_QuantizedDenseMethod):
    """QSGD dense FedAvg implementation on the method API."""

    name = 'qsgd_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='QSGD quantized dense upload FedAvg')

    def client_update(
        self,
        *,
        local_state,
        global_state,
        common: dict[str, Any],
        evaluation_kwargs: dict[str, Any],
        result_cls,
        client,
        round_index: int,
        **_: Any,
    ):
        """Return a QSGD dense update payload from one local model state."""

        levels = int(client.config.get('federated', {}).get('qsgd_levels', 127))
        quantized = quantize_qsgd_state_update(
            subtract_state(local_state, global_state),
            levels=levels,
            generator=client._upload_quantization_generator(round_index),
        )
        return result_cls(
            **common,
            state=quantized,
            **evaluation_kwargs,
            upload_bytes=state_num_bytes(quantized),
            upload_parameters=state_num_parameters(quantized),
            parameter_upload_bytes=state_num_bytes(quantized),
            parameter_upload_parameters=state_num_parameters(quantized),
            transport_upload_bytes=state_num_bytes(quantized),
            payload_kind='qsgd_update',
            compressor=f'qsgd_{levels}_levels',
        )

    def aggregate(self, *, server, results, round_base_state=None, **_: Any) -> list[float]:
        """Aggregate dequantized QSGD uploads with standard FedAvg weighting."""

        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        levels = int(server.config.get('federated', {}).get('qsgd_levels', 127))
        dense_updates = [dequantize_qsgd_state_update(result.state, levels=levels) for result in results]
        averaged_update = average_states(dense_updates, sample_weights)
        server.global_state = add_update(server.global_state, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        return weights

    def extract_attack_payload(self, *, result, server=None, **kwargs: Any):
        """Expose the dequantized QSGD payload to the server attacker."""

        if result.state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        levels = 127 if server is None else int(server.config.get('federated', {}).get('qsgd_levels', 127))
        return dequantize_qsgd_state_update(result.state, levels=levels)
