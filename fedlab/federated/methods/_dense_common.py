"""Shared helpers for dense FedAvg-style methods."""

from __future__ import annotations

from typing import Any

from fedlab.federated.methods.base import FederatedMethod
from fedlab.federated.protocol import build_upload_payload_state, derive_update_from_upload_payload, resolve_upload_mode
from fedlab.utils.serialization import state_num_bytes, state_num_parameters


class DenseFedAvgMethodBase(FederatedMethod):
    """Shared dense-update behavior for FedAvg-style algorithms."""

    def client_update(
        self,
        *,
        local_state,
        global_state,
        received_global_state,
        common: dict[str, Any],
        evaluation_kwargs: dict[str, Any],
        result_cls,
        client=None,
        **_: Any,
    ):
        """Return a dense semantic payload from one local model state."""

        upload_mode = resolve_upload_mode(client.config if client is not None else {})
        base_state = received_global_state if received_global_state is not None else global_state
        payload_state = build_upload_payload_state(local_state, base_state, upload_mode)
        payload_bytes = state_num_bytes(payload_state)
        payload_parameters = state_num_parameters(payload_state)
        return result_cls(
            **common,
            aggregation_state=payload_state,
            **evaluation_kwargs,
            upload_mode=upload_mode,
            upload_bytes=payload_bytes,
            upload_parameters=payload_parameters,
            parameter_upload_bytes=payload_bytes,
            parameter_upload_parameters=payload_parameters,
            transport_upload_bytes=payload_bytes,
            aggregation_payload_kind='dense_model' if upload_mode == 'model' else 'dense_update',
        )

    def extract_attack_payload(self, *, result, clone_state, server=None, round_base_state=None, round_index: int = 0, round_context=None, **_: Any):
        """Expose the dense uploaded client payload as a derived update to the server attacker."""

        if result.aggregation_state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        payload_state = clone_state(result.aggregation_state)
        if result.upload_mode == 'model':
            if server is None or round_base_state is None:
                raise ValueError('Model-mode dense attack extraction requires server context and round base state')
            base_state = server.method.reconstruct_received_global_state(
                server=server,
                global_state=round_base_state,
                client_id=result.client_id,
                round_index=round_index,
                round_context=round_context or {},
            )
            return derive_update_from_upload_payload(payload_state, base_state, result.upload_mode)
        return payload_state
