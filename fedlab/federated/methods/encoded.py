"""Encoded federated algorithm implementations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from fedlab.federated.methods.base import FederatedMethod, MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.federated.protocol import weighted_protocol_base_state
from fedlab.modeling import build_model
from fedlab.modeling.ega import decode_attack_view_from_mean_difference, decode_mean_encoded_payload, encode_state_update, export_ega_codec_payload, load_ega_codec, load_ega_codec_payload
from fedlab.utils.serialization import add_update, average_states, serialize_trainable_model, serialize_untrainable_model, state_num_bytes, state_num_parameters, subtract_state


def _drop_zero_state(state, tolerance: float = 0.0):
    """Remove tensors whose updates are exactly or effectively zero."""

    threshold = max(0.0, float(tolerance))
    return type(state)((name, tensor) for name, tensor in state.items() if float(tensor.detach().abs().max().item()) > threshold)






def _quantization_generator(config: dict[str, Any], round_index: int, client_id: str) -> torch.Generator | None:
    """Return the deterministic quantization generator used by EGA upload quantization."""

    seed = config.get("federated", {}).get("quantization_seed")
    if seed is None:
        return None
    offset = sum(ord(char) for char in client_id)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + round_index * 1000 + offset)
    return generator


def _resolve_ega_total_clients(config: dict[str, Any]) -> int:
    """Resolve the configured EGA client count, allowing an explicit auto mode."""

    ega_cfg = config.get("ega", {})
    configured = ega_cfg.get("num_clients", "auto")
    if str(configured).strip().lower() == "auto":
        return int(len(config.get("data", {}).get("clients", [])) or 1)
    return max(1, int(configured))


def _ensure_client_ega_codec(*, client: Any, round_context: dict[str, Any]) -> None:
    """Initialize one client-side EGA codec from the server bootstrap payload."""

    if getattr(client, "ega_codec", None) is not None:
        return
    codec_payload = round_context.get("ega_codec_payload")
    if codec_payload is None:
        raise RuntimeError("EGA client is missing the server codec bootstrap payload")
    client.ega_codec = load_ega_codec_payload(
        client.config,
        codec_payload,
        device=client.device,
        num_clients=client.total_clients,
    )



@federated_method("ega_fedavg", compressed=False, description="Encoded gradient aggregation FedAvg variant")
class EGAFedAvgMethod(FederatedMethod):
    """Concrete EGA FedAvg implementation on the method API."""

    name = "ega_fedavg"

    capabilities = MethodCapabilities(compressed=False, implemented=True, description="Encoded gradient aggregation FedAvg variant")
    config_spec = MethodConfigSpec(
        federated_keys=frozenset({"quantization_seed"}),
        root_blocks=frozenset({"ega"}),
    )

    def configure_client(self, client: Any) -> None:
        """Initialize client-side EGA bookkeeping; codec bootstrap comes from the server."""

        client.ega_codec = None
        template = build_model(client.config)
        client.ega_trainable_keys = tuple(serialize_trainable_model(template).keys())
        client.ega_residual = None

    def configure_server(self, server: Any) -> None:
        """Initialize the server-owned EGA bookkeeping and optionally defer codec loading."""

        ega_cfg = server.config.get("ega", {})
        total_clients = _resolve_ega_total_clients(server.config)
        server.ega_total_clients = total_clients
        server.ega_codec = None
        server.ega_codec_bootstrap_payload = None
        server.ega_codec_bootstrap_pending = False
        server.ega_normalization = float(ega_cfg.get("initial_normalization", ega_cfg.get("normalization", 1.0)))
        if bool(server.config.get("grpc", {}).get("defer_server_runtime_init", False)):
            return
        self.initialize_server_runtime(server)

    def initialize_server_runtime(self, server: Any) -> None:
        """Load or pretrain the server-owned EGA codec when the transport is ready."""

        if server.ega_codec is not None:
            return
        server.ega_codec = load_ega_codec(
            server.config,
            device=server.device,
            num_clients=server.ega_total_clients,
            allow_pretrain=True,
        )
        server.ega_codec_bootstrap_payload = export_ega_codec_payload(
            server.ega_codec,
            config=server.config,
            num_clients=server.ega_total_clients,
        )
        server.ega_codec_bootstrap_pending = True

    def server_ready(self, server: Any) -> bool:
        """Return whether the EGA codec is ready for round-0 communication."""

        return server.ega_codec is not None

    def build_round_context(self, server: Any) -> dict[str, Any]:
        """Broadcast the current EGA normalization and first-round codec bootstrap."""

        if server.ega_normalization is None:
            return {}
        payload = {"ega_normalization": float(server.ega_normalization)}
        if getattr(server, "ega_codec_bootstrap_pending", False):
            payload["ega_codec_payload"] = server.ega_codec_bootstrap_payload
        return payload

    def client_update(
        self,
        *,
        client,
        model,
        local_state,
        global_state,
        received_global_state,
        common: dict[str, Any],
        evaluation_kwargs: dict[str, Any],
        result_cls,
        round_context: dict[str, Any],
        round_index: int,
        **_: Any,
    ):
        """Return an encoded trainable update plus exact buffer payload."""

        _ensure_client_ega_codec(client=client, round_context=round_context)
        if client.ega_codec is None:
            raise RuntimeError("EGA codec was not initialized on the client")
        ega_cfg = client.config.get("ega", {})
        trainable_state = serialize_trainable_model(model)
        untrainable_state = serialize_untrainable_model(model)
        reference_state = received_global_state if received_global_state is not None else global_state
        global_trainable_state = type(reference_state)((name, reference_state[name]) for name in trainable_state.keys())
        global_untrainable_state = type(reference_state)((name, reference_state[name]) for name in untrainable_state.keys())
        trainable_update = subtract_state(trainable_state, global_trainable_state)
        buffer_update = _drop_zero_state(
            subtract_state(untrainable_state, global_untrainable_state),
            tolerance=float(ega_cfg.get("buffer_tolerance", 0.0)),
        )
        effective_update = trainable_update
        if bool(ega_cfg.get("error_feedback", False)) and getattr(client, "ega_residual", None) is not None:
            effective_update = add_update(trainable_update, client.ega_residual)
        contribution_scale = float(client._loader_num_samples(client.train_loader)) / float(max(client.total_train_samples, 1))
        contribution_scale *= float(client.total_clients)
        normalization = float(
            round_context.get(
                "ega_normalization",
                ega_cfg.get("initial_normalization", ega_cfg.get("normalization", 1.0)),
            )
        )
        min_normalization = float(ega_cfg.get("min_normalization", 1e-6))
        payload = encode_state_update(
            effective_update,
            client.ega_codec,
            quantization_level=int(ega_cfg.get("quantization_level", 64)),
            normalization=max(normalization, min_normalization),
            block_size=int(ega_cfg.get("block_size", 256)),
            contribution_scale=contribution_scale,
            generator=client._upload_quantization_generator(round_index),
            encoded_dtype=str(ega_cfg.get("encoded_dtype", "float32")),
            encoded_stochastic_rounding=bool(ega_cfg.get("encoded_stochastic_rounding", False)),
            encoded_noise_std=float(ega_cfg.get("encoded_noise_std", 0.0)),
        )
        if bool(ega_cfg.get("error_feedback", False)):
            reconstructed_effective = decode_mean_encoded_payload([payload], client.ega_codec)
            scale = float(contribution_scale) if abs(float(contribution_scale)) > 1e-12 else 1.0
            approx_raw = type(effective_update)((name, tensor / scale) for name, tensor in reconstructed_effective.items())
            client.ega_residual = subtract_state(effective_update, approx_raw)
        else:
            client.ega_residual = None
        buffer_bytes = state_num_bytes(buffer_update)
        buffer_parameters = state_num_parameters(buffer_update)
        compressor = f"ega_b{payload.block_size}_h{payload.encoded_dim}_s{payload.quantization_level}_{payload.encoded_dtype}"
        return result_cls(
            **common,
            aggregation_state=buffer_update,
            ega_payload=payload,
            **evaluation_kwargs,
            upload_bytes=payload.algorithm_num_bytes + buffer_bytes,
            upload_parameters=payload.algorithm_num_parameters + buffer_parameters,
            parameter_upload_bytes=payload.algorithm_num_bytes + buffer_bytes,
            parameter_upload_parameters=payload.algorithm_num_parameters + buffer_parameters,
            transport_upload_bytes=payload.algorithm_num_bytes + buffer_bytes,
            aggregation_payload_kind="ega_encoded_update",
            compressor=compressor,
        )

    def _protocol_base_state(self, *, server, results, round_base_state, round_index: int):
        """Return the weighted protocol base visible to clients before applying uploaded updates."""

        if round_base_state is None:
            return server.global_state
        return weighted_protocol_base_state(server, results, round_base_state, round_index, {})

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, **_: Any) -> list[float]:
        """Aggregate encoded client updates with the server-side EGA codec."""

        if server.ega_codec is None:
            raise RuntimeError("EGA codec is not initialized on the server")
        server.ega_codec_bootstrap_pending = False
        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        payloads = [result.ega_payload for result in results]
        if any(payload is None for payload in payloads):
            raise ValueError("EGA aggregation requires ega_payload from every client")
        averaged_update = decode_mean_encoded_payload(payloads, server.ega_codec)
        if any(result.aggregation_state is not None for result in results):
            for result, weight in zip(results, weights):
                if result.aggregation_state is None:
                    continue
                for name, tensor in result.aggregation_state.items():
                    averaged_update[name] = averaged_update.get(name, torch.zeros_like(tensor)) + tensor * weight
        full_update = type(server.global_state)((name, averaged_update.get(name, torch.zeros_like(tensor))) for name, tensor in server.global_state.items())
        protocol_base_state = self._protocol_base_state(
            server=server,
            results=results,
            round_base_state=round_base_state,
            round_index=round_index,
        )
        server.global_state = add_update(protocol_base_state, full_update)
        ega_cfg = server.config.get("ega", {})
        strategy = str(ega_cfg.get("normalization_strategy", "fixed")).lower()
        min_norm = float(ega_cfg.get("min_normalization", 1e-6))
        if strategy == "previous_round_max_abs":
            server.ega_normalization = max(
                min_norm,
                max(float(tensor.abs().max().item()) for tensor in averaged_update.values()),
            )
        elif strategy in {"reported_client_max_abs", "client_reported_max_abs"}:
            server.ega_normalization = max(min_norm, max(float(payload.observed_update_absmax) for payload in payloads))
        elif strategy in {"ema_reported_client_max_abs", "ema_client_reported_max_abs"}:
            observed = max(float(payload.observed_update_absmax) for payload in payloads)
            decay = float(ega_cfg.get("normalization_ema", 0.9))
            previous = float(server.ega_normalization if server.ega_normalization is not None else observed)
            server.ega_normalization = max(min_norm, decay * previous + (1.0 - decay) * observed)
        return weights

    def extract_attack_payload(self, *, result, results, server=None, **_: Any):
        """Return the honest-but-curious server attack view for one EGA client."""

        if server is None:
            raise ValueError("EGA attack extraction requires server context")
        payloads = [item.ega_payload for item in results]
        if any(payload is None for payload in payloads):
            raise ValueError("EGA attack extraction requires payloads from every client")
        target_index = next(index for index, item in enumerate(results) if item.client_id == result.client_id)
        attack_view = decode_attack_view_from_mean_difference(payloads, target_index, server.ega_codec)
        target_result = results[target_index]
        if target_result.aggregation_state is not None:
            attack_view.update(target_result.aggregation_state)
        return attack_view
