"""gRPC transport for multi-process federated training.

The service intentionally serializes Python payloads with pickle over generic
unary RPC methods. This keeps the framework self-contained and avoids a proto
generation step while still using gRPC as the transport layer.
"""

from __future__ import annotations

import pickle
from concurrent import futures
from typing import Any, Callable

from loguru import logger


try:  # pragma: no cover - optional dependency
    import grpc
except Exception:  # pragma: no cover
    grpc = None


def _dumps(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _loads(value: bytes) -> Any:
    return pickle.loads(value)


class FederatedRpcServer:
    """Minimal gRPC server exposing ``GetGlobal`` and ``SubmitUpdate``."""

    def __init__(self, address: str, get_global: Callable[[], Any], submit_update: Callable[[Any], Any]):
        if grpc is None:
            raise RuntimeError("grpcio is not installed")
        self.address = address
        self.get_global = get_global
        self.submit_update = submit_update
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
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
        }
        self.server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler("FederatedService", handlers),))

    def start(self) -> None:
        self.server.add_insecure_port(self.address)
        self.server.start()
        logger.info("gRPC server listening on {}", self.address)

    def wait(self) -> None:
        self.server.wait_for_termination()

    def stop(self, grace: int = 0) -> None:
        self.server.stop(grace)


class FederatedRpcClient:
    """Client helper for the generic gRPC service."""

    def __init__(self, address: str):
        if grpc is None:
            raise RuntimeError("grpcio is not installed")
        self.channel = grpc.insecure_channel(address)
        self.get_global_rpc = self.channel.unary_unary("/FederatedService/GetGlobal")
        self.submit_update_rpc = self.channel.unary_unary("/FederatedService/SubmitUpdate")

    def get_global(self) -> Any:
        return _loads(self.get_global_rpc(b""))

    def submit_update(self, update: Any) -> Any:
        return _loads(self.submit_update_rpc(_dumps(update)))

