"""gRPC transport for multi-process federated training.

The service intentionally serializes Python payloads with pickle over generic
unary RPC methods. This keeps the framework self-contained and avoids a proto
generation step while still using gRPC as the transport layer.
"""

from __future__ import annotations

import pickle
from concurrent import futures
from dataclasses import asdict, dataclass
from typing import Any, Callable

from loguru import logger


try:  # pragma: no cover - optional dependency
    import grpc
except Exception:  # pragma: no cover
    grpc = None


@dataclass
class RpcTransportStats:
    """Serialized request and response byte counters for one RPC client."""

    sent_bytes: int = 0
    received_bytes: int = 0
    request_count: int = 0
    response_count: int = 0


def _dumps(value: Any) -> bytes:
    """Serialize an RPC payload with pickle."""

    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _loads(value: bytes) -> Any:
    """Deserialize an RPC payload created by ``_dumps``."""

    return pickle.loads(value)


class FederatedRpcServer:
    """Minimal gRPC server exposing ``GetGlobal`` and ``SubmitUpdate``."""

    def __init__(
        self,
        address: str,
        get_global: Callable[[], Any],
        submit_update: Callable[[Any], Any],
        ack_stop: Callable[[Any], Any],
        max_message_length: int = 256 * 1024 * 1024,
    ):
        """Create a generic gRPC server around coordinator callbacks."""

        if grpc is None:
            raise RuntimeError("grpcio is not installed")
        self.address = address
        self.get_global = get_global
        self.submit_update = submit_update
        self.ack_stop = ack_stop
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=8),
            options=[
                ("grpc.max_send_message_length", int(max_message_length)),
                ("grpc.max_receive_message_length", int(max_message_length)),
            ],
        )
        handlers = {
            "GetGlobal": grpc.unary_unary_rpc_method_handler(
                lambda request, context: _dumps(self.get_global()),
                request_deserializer=lambda raw: raw,
                response_serializer=lambda raw: raw,
            ),
            "SubmitUpdate": grpc.unary_unary_rpc_method_handler(
                lambda request, context: _dumps(self.submit_update(_loads(request))),
                request_deserializer=lambda raw: raw,
                response_serializer=lambda raw: raw,
            ),
            "AckStop": grpc.unary_unary_rpc_method_handler(
                lambda request, context: _dumps(self.ack_stop(_loads(request))),
                request_deserializer=lambda raw: raw,
                response_serializer=lambda raw: raw,
            ),
        }
        self.server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler("FederatedService", handlers),))

    def start(self) -> None:
        """Start listening for federated RPC requests."""

        self.server.add_insecure_port(self.address)
        self.server.start()
        logger.info("gRPC server listening on {}", self.address)

    def wait(self) -> None:
        """Block until the gRPC server terminates."""

        self.server.wait_for_termination()

    def stop(self, grace: int = 0) -> None:
        """Stop the gRPC server after the optional grace period."""

        self.server.stop(grace)


class FederatedRpcClient:
    """Client helper for the generic gRPC service."""

    def __init__(self, address: str, max_message_length: int = 256 * 1024 * 1024):
        """Create RPC stubs for a federated server address."""

        if grpc is None:
            raise RuntimeError("grpcio is not installed")
        self.channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", int(max_message_length)),
                ("grpc.max_receive_message_length", int(max_message_length)),
            ],
        )
        self.get_global_rpc = self.channel.unary_unary("/FederatedService/GetGlobal")
        self.submit_update_rpc = self.channel.unary_unary("/FederatedService/SubmitUpdate")
        self.ack_stop_rpc = self.channel.unary_unary("/FederatedService/AckStop")
        self.stats = RpcTransportStats()

    def _record_exchange(self, request_bytes: bytes, response_bytes: bytes) -> None:
        """Accumulate serialized request and response byte counters."""

        self.stats.sent_bytes += len(request_bytes)
        self.stats.received_bytes += len(response_bytes)
        self.stats.request_count += 1
        self.stats.response_count += 1

    def snapshot_counters(self) -> dict[str, int]:
        """Return a snapshot of the serialized transport counters."""

        return asdict(self.stats)

    def get_global(self) -> Any:
        """Fetch the current global model payload from the server."""

        request_bytes = b""
        response_bytes = self.get_global_rpc(request_bytes)
        self._record_exchange(request_bytes, response_bytes)
        return _loads(response_bytes)

    def ack_stop(self, payload: Any) -> Any:
        """Acknowledge that one client observed the server stop signal."""

        request_bytes = _dumps(payload)
        response_bytes = self.ack_stop_rpc(request_bytes)
        self._record_exchange(request_bytes, response_bytes)
        return _loads(response_bytes)

    def submit_update(self, update: Any) -> Any:
        """Submit one client update payload to the server."""

        request_bytes = _dumps(update)
        response_bytes = self.submit_update_rpc(request_bytes)
        self._record_exchange(request_bytes, response_bytes)
        return _loads(response_bytes)
