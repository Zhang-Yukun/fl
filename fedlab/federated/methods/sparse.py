"""Sparse federated algorithm implementations."""

from __future__ import annotations

from typing import Any

from fedlab.federated.methods.base import FederatedMethod, MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.federated.protocol import resolve_upload_mode, weighted_protocol_base_state
from fedlab.utils.serialization import (
    add_update,
    compress_randomk,
    compress_topk,
    decompress_topk,
    privatize_state_update,
    serialize_trainable_model,
    serialize_untrainable_model,
    state_num_bytes,
    state_num_parameters,
    subtract_state,
)


class _SparseFedAvgMethod(FederatedMethod):
    """Shared sparse-upload behavior for Top-k and Random-k FedAvg variants."""

    def _split_updates(self, *, model, base_state):
        """Return trainable sparse candidates plus dense buffer updates."""

        trainable_state = serialize_trainable_model(model)
        untrainable_state = serialize_untrainable_model(model)
        global_trainable_state = type(base_state)((name, base_state[name]) for name in trainable_state.keys())
        global_untrainable_state = type(base_state)((name, base_state[name]) for name in untrainable_state.keys())
        return (
            subtract_state(trainable_state, global_trainable_state),
            subtract_state(untrainable_state, global_untrainable_state),
        )

    def _build_sparse_result(
        self,
        *,
        sparse_update,
        buffer_update,
        common: dict[str, Any],
        evaluation_kwargs: dict[str, Any],
        result_cls,
        aggregation_payload_kind: str,
        compressor: str,
        privacy_clip_norm: float = 0.0,
        privacy_noise_multiplier: float = 0.0,
    ):
        """Construct a sparse client result with optional dense buffer updates."""

        dense_buffer_bytes = state_num_bytes(buffer_update)
        dense_buffer_parameters = state_num_parameters(buffer_update)
        return result_cls(
            **common,
            aggregation_state=buffer_update,
            sparse_update=sparse_update,
            **evaluation_kwargs,
            upload_bytes=sparse_update.nbytes + dense_buffer_bytes,
            upload_parameters=sparse_update.values.numel() + dense_buffer_parameters,
            parameter_upload_bytes=sparse_update.nbytes + dense_buffer_bytes,
            parameter_upload_parameters=sparse_update.values.numel() + dense_buffer_parameters,
            transport_upload_bytes=sparse_update.nbytes + dense_buffer_bytes,
            aggregation_payload_kind=aggregation_payload_kind,
            compressor=compressor,
            privacy_clip_norm=privacy_clip_norm,
            privacy_noise_multiplier=privacy_noise_multiplier,
        )

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: Any) -> list[float]:
        """Aggregate sparse client updates for compressed FedAvg variants."""

        weights = [result.num_samples for result in results]
        total = float(sum(weights))
        update = None
        for result, weight in zip(results, weights):
            dense = decompress_topk(result.sparse_update)
            if result.aggregation_state is not None:
                dense.update(result.aggregation_state)
            scaled = {name: tensor * (weight / total) for name, tensor in dense.items()}
            update = scaled if update is None else {name: update[name] + scaled[name] for name in scaled}
        protocol_base_state = weighted_protocol_base_state(server, results, round_base_state, round_index, round_context or {})
        server.global_state = add_update(protocol_base_state, update)
        server._update_oracle_evaluation_state(round_base_state, results, weights)
        return [weight / total for weight in weights]

    def extract_attack_payload(self, *, result, clone_state, **_: Any):
        """Expose the decompressed sparse update plus dense buffers to the attacker."""

        payload = decompress_topk(result.sparse_update)
        if result.aggregation_state is not None:
            payload.update(clone_state(result.aggregation_state))
        return payload


@federated_method('compressed_fedavg', compressed=True, description='Legacy top-k sparse FedAvg alias')
class CompressedFedAvgMethod(_SparseFedAvgMethod):
    """Concrete legacy compressed FedAvg alias using Top-k sparse updates."""

    name = 'compressed_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=True, description='Legacy top-k sparse FedAvg alias')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'topk_fraction'}))

    def client_update(self, *, model, global_state, received_global_state=None, common: dict[str, Any], evaluation_kwargs: dict[str, Any], result_cls, client=None, **_: Any):
        """Return a Top-k sparse payload from one local model state."""

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


@federated_method('sparse_fedavg', compressed=True, description='Top-k sparse FedAvg baseline')
class SparseFedAvgMethod(_SparseFedAvgMethod):
    """Concrete Top-k sparse FedAvg baseline implementation."""

    name = 'sparse_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=True, description='Top-k sparse FedAvg baseline')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'topk_fraction'}))

    def client_update(self, *, model, global_state, received_global_state=None, common: dict[str, Any], evaluation_kwargs: dict[str, Any], result_cls, client=None, **_: Any):
        """Return a Top-k sparse payload from one local model state."""

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


@federated_method('dp_topk_fedavg', compressed=True, description='Top-k sparse FedAvg with DP preprocessing')
class DpTopkFedAvgMethod(_SparseFedAvgMethod):
    """Concrete DP Top-k FedAvg implementation."""

    name = 'dp_topk_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=True, description='Top-k sparse FedAvg with DP preprocessing')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'topk_fraction'}), uses_privacy_block=True)

    def client_update(self, *, model, global_state, received_global_state=None, common: dict[str, Any], evaluation_kwargs: dict[str, Any], result_cls, client, **_: Any):
        """Return a DP-processed Top-k sparse payload from one local model state."""

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


@federated_method('randomk_fedavg', compressed=True, description='Random-k sparse FedAvg baseline')
class RandomkFedAvgMethod(_SparseFedAvgMethod):
    """Concrete Random-k sparse FedAvg implementation."""

    name = 'randomk_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=True, description='Random-k sparse FedAvg baseline')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'topk_fraction', 'randomk_seed'}))

    def client_update(self, *, model, global_state, received_global_state=None, common: dict[str, Any], evaluation_kwargs: dict[str, Any], result_cls, client, round_index: int, **_: Any):
        """Return a Random-k sparse payload from one local model state."""

        upload_mode = resolve_upload_mode(client.config)
        if upload_mode != 'update':
            raise ValueError('Sparse upload methods only support transport.upload_mode=update')
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


@federated_method('soteriafl', compressed=True, description='Private random-k sparse upload baseline')
class SoteriaFLMethod(_SparseFedAvgMethod):
    """Concrete private Random-k sparse upload implementation."""

    name = 'soteriafl'
    capabilities = MethodCapabilities(compressed=True, implemented=True, description='Private random-k sparse upload baseline')
    config_spec = MethodConfigSpec(federated_keys=frozenset({'topk_fraction', 'randomk_seed'}), uses_privacy_block=True)

    def client_update(self, *, model, global_state, received_global_state=None, common: dict[str, Any], evaluation_kwargs: dict[str, Any], result_cls, client, round_index: int, **_: Any):
        """Return a DP-processed Random-k sparse payload from one local model state."""

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
        sparse = compress_randomk(update, fraction, generator=client._randomk_generator(round_index))
        return self._build_sparse_result(
            sparse_update=sparse,
            buffer_update=buffer_update,
            common=common,
            evaluation_kwargs=evaluation_kwargs,
            result_cls=result_cls,
            aggregation_payload_kind='soteriafl_randomk_dp_update',
            compressor='randomk_unbiased',
            privacy_clip_norm=privacy_clip_norm,
            privacy_noise_multiplier=privacy_noise_multiplier,
        )
