"""High-level training algorithms: centralized, FedAvg, and compressed FedAvg."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from federated_ts.attacks import attack_success_rate, dlg_attack, idlg_attack
from federated_ts.client import FederatedClient
from federated_ts.data import build_federated_loaders
from federated_ts.logging_utils import setup_logging
from federated_ts.models import build_model
from federated_ts.server import EarlyStopper, FederatedServer
from federated_ts.tracking import Tracker
from federated_ts.training import evaluate, train_one_epoch


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    return torch.device(requested)


def run_centralized(config: dict[str, Any]) -> dict[str, float]:
    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir, config.get("runtime", {}).get("log_level", "INFO"))
    device = resolve_device(config)
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("lr", 1e-3)))
    stopper = EarlyStopper(int(config["training"].get("patience", 5)))
    for epoch in range(int(config["training"].get("epochs", 10))):
        loss = sum(train_one_epoch(model, loader, optimizer, device) for loader in train_loaders.values()) / len(train_loaders)
        metrics = evaluate(model, val_loader, device)
        logger.info("Centralized epoch {} loss={:.6f} val_mse={:.6f}", epoch, loss, metrics["mse"])
        if stopper.update(metrics["mse"]):
            break
    torch.save(model.state_dict(), output_dir / "centralized_model.pt")
    return evaluate(model, test_loader, device)


def run_federated(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir, config.get("runtime", {}).get("log_level", "INFO"))
    tracker = Tracker(config)
    device = resolve_device(config)
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    server = FederatedServer(config, val_loader, test_loader, device)
    clients = [FederatedClient(client_id, loader, config, device) for client_id, loader in train_loaders.items()]
    compressed = str(config["federated"].get("algorithm", "fedavg")).lower() in {"compressed_fedavg", "sparse_fedavg"}
    stopper = EarlyStopper(int(config["training"].get("patience", 5)), float(config["training"].get("min_delta", 0.0)))
    max_rounds = int(config["federated"].get("rounds", 20))
    attack_results = []
    for round_index in range(max_rounds):
        results = [client.train(server.global_state, compressed=compressed) for client in clients]
        if compressed:
            server.aggregate_sparse(results)
        else:
            server.aggregate_dense(results)
        metrics = server.evaluate_global()
        record = server.record_round(round_index, results, metrics)
        tracker.log(record.__dict__, step=round_index)
        if config.get("attack", {}).get("enabled", True):
            grads, real_x, real_y = clients[0].gradient_sample(server.global_state)
            attack_results.extend(
                [
                    dlg_attack(config, server.global_state, grads, real_x, real_y, device),
                    idlg_attack(config, server.global_state, grads, real_x, real_y, device),
                ]
            )
        if stopper.update(metrics["mse"]):
            logger.info("Early stopping at round {}", round_index)
            break
    test_metrics = server.test_global()
    server.save(output_dir, config)
    tracker.log({f"test/{key}": value for key, value in test_metrics.items()})
    tracker.finish()
    summary = {
        "test": test_metrics,
        "rounds": len(server.history),
        "last_communication_ratio": server.history[-1].communication_ratio if server.history else 0.0,
        "attack_success_rate": attack_success_rate(attack_results),
    }
    logger.info("Finished experiment: {}", summary)
    return summary

