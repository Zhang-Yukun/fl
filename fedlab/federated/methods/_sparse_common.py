"""Shared helpers for sparse federated methods."""

from __future__ import annotations

from fedlab.federated.methods.base import FederatedMethod
from fedlab.federated.protocol import weighted_protocol_base_state
from fedlab.utils.serialization import (
    add_update,
    decompress_topk,
    serialize_trainable_model,
    serialize_untrainable_model,
    state_num_bytes,
    state_num_parameters,
    subtract_state,
)


class SparseFedAvgMethodBase(FederatedMethod):
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
        common: dict,
        evaluation_kwargs: dict,
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

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: object) -> list[float]:
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

    def extract_attack_payload(self, *, result, clone_state, **_: object):
        """Expose the decompressed sparse update plus dense buffers to the attacker."""

        payload = decompress_topk(result.sparse_update)
        if result.aggregation_state is not None:
            payload.update(clone_state(result.aggregation_state))
        return payload
