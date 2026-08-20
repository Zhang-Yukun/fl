"""Transport-semantic helpers for federated upload and download modes."""

from __future__ import annotations

from typing import Any

from fedlab.utils.serialization import StateDict, add_update, average_states, subtract_state

UPLOAD_MODES = {"update", "model"}
DOWNLOAD_MODES = {"model", "update"}
UPLOAD_MODEL_UNSUPPORTED = {"compressed_fedavg", "sparse_fedavg", "dp_topk_fedavg", "randomk_fedavg", "soteriafl", "secure_quantized_fedavg", "sign_fedavg", "qsgd_fedavg", "ega_fedavg"}
DOWNLOAD_UPDATE_UNSUPPORTED: set[str] = set()


def resolve_upload_mode(config: dict[str, Any]) -> str:
    """Return the configured upload transport semantic mode."""

    mode = str(config.get("transport", {}).get("upload_mode", "update")).lower()
    if mode not in UPLOAD_MODES:
        raise ValueError(f"Unsupported transport.upload_mode: {mode}")
    return mode


def resolve_download_mode(config: dict[str, Any]) -> str:
    """Return the configured download transport semantic mode."""

    mode = str(config.get("transport", {}).get("download_mode", "model")).lower()
    if mode not in DOWNLOAD_MODES:
        raise ValueError(f"Unsupported transport.download_mode: {mode}")
    return mode


def validate_transport_modes(config: dict[str, Any], transport_backend: str = "local") -> None:
    """Validate algorithm/mode combinations supported by the current runtime."""

    algorithm = str(config.get("federated", {}).get("algorithm", "fedavg")).lower()
    upload_mode = resolve_upload_mode(config)
    download_mode = resolve_download_mode(config)
    if upload_mode == "model" and algorithm in UPLOAD_MODEL_UNSUPPORTED:
        raise ValueError(f"Algorithm {algorithm} does not support transport.upload_mode=model")
    if download_mode == "update" and algorithm in DOWNLOAD_UPDATE_UNSUPPORTED:
        raise ValueError(f"Algorithm {algorithm} does not support transport.download_mode=update")
    if transport_backend == "grpc" and download_mode == "update":
        raise ValueError("gRPC transport does not yet support transport.download_mode=update")


def build_upload_payload_state(local_state: StateDict, base_state: StateDict, upload_mode: str) -> StateDict:
    """Return the semantic payload state to upload before optional compression."""

    if upload_mode == "model":
        return local_state
    return subtract_state(local_state, base_state)


def derive_update_from_upload_payload(payload_state: StateDict, base_state: StateDict, upload_mode: str) -> StateDict:
    """Return the canonical aggregation update from one uploaded semantic payload."""

    if upload_mode == "model":
        return subtract_state(payload_state, base_state)
    return payload_state


def build_download_payload_state(target_model_state: StateDict, base_state: StateDict, download_mode: str) -> StateDict:
    """Return the semantic payload state to download before client-side reconstruction."""

    if download_mode == "model":
        return target_model_state
    return subtract_state(target_model_state, base_state)


def reconstruct_model_from_download_payload(payload_state: StateDict, base_state: StateDict, download_mode: str) -> StateDict:
    """Return the client-visible model reconstructed from one download payload."""

    if download_mode == "model":
        return payload_state
    return add_update(base_state, payload_state)


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
