"""Transport-semantic helpers for the fixed federated payload semantics."""

from __future__ import annotations

from typing import Any

from fedlab.utils.serialization import StateDict, average_states, subtract_state


def validate_transport_modes(config: dict[str, Any], transport_backend: str = "local") -> None:
    """Validate the fixed transport semantics.

    Uploads are always model updates and downloads are always full models.
    The helper remains as a compatibility hook for local and gRPC startup
    validation.
    """

    del config, transport_backend


def build_upload_payload_state(local_state: StateDict, base_state: StateDict) -> StateDict:
    """Return the canonical uploaded update payload."""

    return subtract_state(local_state, base_state)


def derive_update_from_upload_payload(payload_state: StateDict, base_state: StateDict | None = None) -> StateDict:
    """Return the canonical aggregation update from one uploaded payload."""

    del base_state
    return payload_state


def build_download_payload_state(target_model_state: StateDict, base_state: StateDict | None = None) -> StateDict:
    """Return the canonical downloaded model payload."""

    del base_state
    return target_model_state


def reconstruct_model_from_download_payload(payload_state: StateDict, base_state: StateDict | None = None) -> StateDict:
    """Return the client-visible model reconstructed from one download payload."""

    del base_state
    return payload_state


def weighted_protocol_base_state(server: Any, results, round_base_state: StateDict, round_index: int, round_context: dict[str, Any] | None = None) -> StateDict:
    """Return the weighted average of the client-visible models at this round start."""

    round_context = round_context or {}
    states = [
        server.method.reconstruct_received_global_state(
            server=server,
            global_state=round_base_state,
            client_id=result.client_id,
            round_index=round_index,
            round_context=round_context,
        )
        for result in results
    ]
    sample_weights = [result.num_samples for result in results]
    return average_states(states, sample_weights)
