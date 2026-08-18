"""Encoded federated algorithm implementations."""

from __future__ import annotations

from typing import Any

import torch

from fedlab.federated.methods.base import FederatedMethod, MethodCapabilities
from fedlab.federated.methods.registry import federated_method
from fedlab.modeling.ega import decode_attack_view_from_mean_difference, decode_mean_encoded_payload, encode_state_update, load_ega_codec
from fedlab.utils.serialization import add_update, serialize_trainable_model, serialize_untrainable_model, state_num_bytes, state_num_parameters, subtract_state


@federated_method('ega_fedavg', compressed=False, description='Encoded gradient aggregation FedAvg variant')
class EGAFedAvgMethod(FederatedMethod):
    """Concrete EGA FedAvg implementation on the method API."""

    name = 'ega_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Encoded gradient aggregation FedAvg variant')

    def configure_client(self, client: Any) -> None:
        """Initialize the EGA codec on the client."""

        client.ega_codec = load_ega_codec(
            client.config,
            device=client.device,
            num_clients=client.total_clients,
            allow_pretrain=bool(getattr(client, 'allow_ega_pretrain', False)),
        )

    def configure_server(self, server: Any) -> None:
        """Initialize the EGA codec and normalization state on the server."""

        ega_cfg = server.config.get('ega', {})
        total_clients = int(ega_cfg.get('num_clients', len(server.config.get('data', {}).get('clients', [])) or 1))
        server.ega_codec = load_ega_codec(
            server.config,
            device=server.device,
            num_clients=total_clients,
            allow_pretrain=True,
        )
        server.ega_normalization = float(ega_cfg.get('initial_normalization', ega_cfg.get('normalization', 1.0)))

    def build_round_context(self, server: Any) -> dict[str, Any]:
        """Broadcast the current EGA normalization factor to clients."""

        if server.ega_normalization is None:
            return {}
        return {'ega_normalization': float(server.ega_normalization)}

    def client_update(
        self,
        *,
        client,
        model,
        local_state,
        global_state,
        common: dict[str, Any],
        evaluation_kwargs: dict[str, Any],
        result_cls,
        round_context: dict[str, Any],
        round_index: int,
        **_: Any,
    ):
        """Return an encoded trainable update plus exact buffer payload."""

        if client.ega_codec is None:
            raise RuntimeError('EGA codec was not initialized on the client')
        ega_cfg = client.config.get('ega', {})
        trainable_state = serialize_trainable_model(model)
        untrainable_state = serialize_untrainable_model(model)
        global_trainable_state = type(global_state)((name, global_state[name]) for name in trainable_state.keys())
        global_untrainable_state = type(global_state)((name, global_state[name]) for name in untrainable_state.keys())
        trainable_update = subtract_state(trainable_state, global_trainable_state)
        buffer_update = subtract_state(untrainable_state, global_untrainable_state)
        contribution_scale = float(client._loader_num_samples(client.train_loader)) / float(max(client.total_train_samples, 1))
        contribution_scale *= float(client.total_clients)
        normalization = float(
            round_context.get(
                'ega_normalization',
                ega_cfg.get('initial_normalization', ega_cfg.get('normalization', 1.0)),
            )
        )
        payload = encode_state_update(
            trainable_update,
            client.ega_codec,
            quantization_level=int(ega_cfg.get('quantization_level', 64)),
            normalization=max(normalization, float(ega_cfg.get('min_normalization', 1e-6))),
            block_size=int(ega_cfg.get('block_size', 256)),
            contribution_scale=contribution_scale,
            generator=client._upload_quantization_generator(round_index),
        )
        buffer_bytes = state_num_bytes(buffer_update)
        buffer_parameters = state_num_parameters(buffer_update)
        return result_cls(
            **common,
            aggregation_state=buffer_update,
            ega_payload=payload,
            **evaluation_kwargs,
            upload_bytes=payload.nbytes + buffer_bytes,
            upload_parameters=payload.num_parameters + buffer_parameters,
            parameter_upload_bytes=payload.nbytes + buffer_bytes,
            parameter_upload_parameters=payload.num_parameters + buffer_parameters,
            transport_upload_bytes=payload.nbytes + buffer_bytes,
            aggregation_payload_kind='ega_encoded_update',
            compressor=f'ega_b{payload.block_size}_h{payload.encoded_dim}_s{payload.quantization_level}',
        )

    def aggregate(self, *, server, results, round_base_state=None, **_: Any) -> list[float]:
        """Aggregate encoded client updates with the server-side EGA codec."""

        if server.ega_codec is None:
            raise RuntimeError('EGA codec is not initialized on the server')
        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        payloads = [result.ega_payload for result in results]
        if any(payload is None for payload in payloads):
            raise ValueError('EGA aggregation requires ega_payload from every client')
        averaged_update = decode_mean_encoded_payload(payloads, server.ega_codec)
        if any(result.aggregation_state is not None for result in results):
            for result, weight in zip(results, weights):
                if result.aggregation_state is None:
                    continue
                for name, tensor in result.aggregation_state.items():
                    averaged_update[name] = averaged_update.get(name, torch.zeros_like(tensor)) + tensor * weight
        server.global_state = add_update(server.global_state, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        ega_cfg = server.config.get('ega', {})
        if str(ega_cfg.get('normalization_strategy', 'fixed')).lower() == 'previous_round_max_abs':
            server.ega_normalization = max(
                float(ega_cfg.get('min_normalization', 1e-6)),
                max(float(tensor.abs().max().item()) for tensor in averaged_update.values()),
            )
        return weights

    def extract_attack_payload(self, *, result, results, server=None, **_: Any):
        """Return the honest-but-curious server attack view for one EGA client."""

        if server is None:
            raise ValueError('EGA attack extraction requires server context')
        payloads = [item.ega_payload for item in results]
        if any(payload is None for payload in payloads):
            raise ValueError('EGA attack extraction requires payloads from every client')
        target_index = next(index for index, item in enumerate(results) if item.client_id == result.client_id)
        attack_view = decode_attack_view_from_mean_difference(payloads, target_index, server.ega_codec)
        target_result = results[target_index]
        if target_result.aggregation_state is not None:
            attack_view.update(target_result.aggregation_state)
        return attack_view
