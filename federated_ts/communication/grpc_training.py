"""Multi-process federated training built on the generic gRPC transport."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from federated_ts.federated.algorithms import resolve_device
from federated_ts.federated.client import FederatedClient
from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.communication.grpc_service import FederatedRpcClient, FederatedRpcServer
from federated_ts.utils.logging import setup_logging
from federated_ts.federated.server import EarlyStopper, FederatedServer


class GrpcFederatedCoordinator:
    """Server-side coordinator that aggregates updates when all clients report."""

    def __init__(self, config: dict[str, Any]):
        """Initialize server-side state for multi-process training."""

        self.config = config
        self.output_dir = Path(config["experiment"]["output_dir"])
        setup_logging(self.output_dir, config.get("runtime", {}).get("log_level", "INFO"))
        self.device = resolve_device(config)
        train_loaders, val_loader, test_loader = build_federated_loaders(config)
        self.expected_clients = set(train_loaders.keys())
        self.server = FederatedServer(config, val_loader, test_loader, self.device)
        self.compressed = str(config["federated"].get("algorithm", "fedavg")).lower() in {"compressed_fedavg", "sparse_fedavg"}
        self.max_rounds = int(config["federated"].get("rounds", 20))
        self.stopper = EarlyStopper(int(config["training"].get("patience", 5)), float(config["training"].get("min_delta", 0.0)))
        self.round_index = 0
        self.start_time = time.perf_counter()
        self.round_start_time = time.perf_counter()
        self.pending = {}
        self.stopped = False
        self.lock = threading.Lock()

    def get_global(self) -> dict[str, Any]:
        """Return the current global payload for remote clients."""

        with self.lock:
            return {
                "round": self.round_index,
                "state": self.server.global_state,
                "compressed": self.compressed,
                "stop": self.stopped,
            }

    def submit_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Accept one remote client update and aggregate when all clients arrive."""

        with self.lock:
            if self.stopped:
                return {"accepted": False, "stop": True, "round": self.round_index}
            if payload["round"] != self.round_index:
                return {"accepted": False, "stop": False, "round": self.round_index}
            result = payload["result"]
            self.pending[result.client_id] = result
            logger.info("Received update from {} for round {}", result.client_id, self.round_index)
            if self.expected_clients.issubset(self.pending.keys()):
                results = [self.pending[client_id] for client_id in sorted(self.expected_clients)]
                if self.compressed:
                    self.server.aggregate_sparse(results)
                else:
                    self.server.aggregate_dense(results)
                metrics = self.server.evaluate_global()
                self.server.record_round(
                    self.round_index,
                    results,
                    metrics,
                    round_time_seconds=time.perf_counter() - self.round_start_time,
                    elapsed_time_seconds=time.perf_counter() - self.start_time,
                )
                self.pending = {}
                self.round_index += 1
                self.round_start_time = time.perf_counter()
                self.stopped = self.round_index >= self.max_rounds or self.stopper.update(metrics["mse"])
                if self.stopped:
                    self.server.save(self.output_dir, self.config)
                    logger.info("gRPC federated training stopped after {} rounds", self.round_index)
            return {"accepted": True, "stop": self.stopped, "round": self.round_index}


def serve(config: dict[str, Any]) -> None:
    """Start a blocking gRPC federated server."""

    address = config.get("grpc", {}).get("address", "0.0.0.0:50051")
    coordinator = GrpcFederatedCoordinator(config)
    rpc_server = FederatedRpcServer(address, coordinator.get_global, coordinator.submit_update)
    rpc_server.start()
    try:
        while not coordinator.stopped:
            time.sleep(1.0)
    finally:
        rpc_server.stop(0)


def run_client(config: dict[str, Any], client_id: str) -> None:
    """Run a gRPC federated client loop until the server reports stop."""

    address = config.get("grpc", {}).get("server_address", config.get("grpc", {}).get("address", "127.0.0.1:50051"))
    setup_logging(Path(config["experiment"]["output_dir"]) / f"client_{client_id}", config.get("runtime", {}).get("log_level", "INFO"))
    device = resolve_device(config)
    train_loaders, _, _ = build_federated_loaders(config)
    if client_id not in train_loaders:
        raise ValueError(f"Unknown client_id {client_id}; expected one of {sorted(train_loaders)}")
    client = FederatedClient(client_id, train_loaders[client_id], config, device)
    rpc = FederatedRpcClient(address)
    last_submitted = -1
    while True:
        global_payload = rpc.get_global()
        if global_payload["stop"]:
            logger.info("Client {} received stop signal", client_id)
            return
        round_index = global_payload["round"]
        if round_index == last_submitted:
            time.sleep(float(config.get("grpc", {}).get("poll_seconds", 1.0)))
            continue
        result = client.train(global_payload["state"], compressed=global_payload["compressed"])
        response = rpc.submit_update({"round": round_index, "result": result})
        last_submitted = round_index if response.get("accepted") else last_submitted
        if response.get("stop"):
            logger.info("Client {} completed final round {}", client_id, round_index)
            return
