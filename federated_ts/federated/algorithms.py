"""High-level training algorithms: centralized, FedAvg, and compressed FedAvg."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from federated_ts.utils.artifacts import save_experiment_config
from federated_ts.security.attacks import attack_success_rate, dlg_attack, idlg_attack
from federated_ts.federated.client import FederatedClient
from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.utils.logging import setup_logging
from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import serialize_model, state_num_bytes, state_num_parameters
from federated_ts.federated.server import EarlyStopper, FederatedServer, RoundRecord
from federated_ts.utils.tracking import Tracker
from federated_ts.engine.training import evaluate, train_one_epoch


def configure_torch_runtime(config: dict[str, Any]) -> None:
    """Apply CPU thread limits from runtime config before training starts.

    Example:
        ``configure_torch_runtime({"runtime": {"num_threads": 4}})`` limits
        PyTorch intra-op CPU work to four threads.
    """

    runtime_cfg = config.get("runtime", {})
    num_threads = runtime_cfg.get("num_threads")
    interop_threads = runtime_cfg.get("num_interop_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
        logger.info("Set torch num_threads={}", int(num_threads))
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(interop_threads))
            logger.info("Set torch num_interop_threads={}", int(interop_threads))
        except RuntimeError as exc:
            logger.warning("Could not set torch interop threads after runtime start: {}", exc)


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve the configured torch device with a CPU fallback.

    Example:
        ``resolve_device({"runtime": {"device": "cpu"}})`` returns a CPU device.
    """

    requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    return torch.device(requested)


def is_compressed_algorithm(config: dict[str, Any]) -> bool:
    """Return whether the configured FL algorithm compresses client uploads."""

    algorithm = str(config.get("federated", {}).get("algorithm", "fedavg")).lower()
    if algorithm == "fedavg":
        return False
    if algorithm in {"compressed_fedavg", "sparse_fedavg", "soteriafl"}:
        return True
    raise ValueError(f"Unknown federated algorithm: {algorithm}")


def _should_run_attack(config: dict[str, Any], round_index: int, max_rounds: int) -> bool:
    """Return whether attack evaluation should run on this round.

    Example:
        ``frequency_rounds=50`` runs at rounds 0, 50, 100, and the final round.
    """

    attack_cfg = config.get("attack", {})
    if not attack_cfg.get("enabled", True):
        return False
    frequency = int(attack_cfg.get("frequency_rounds", 1))
    return round_index == 0 or round_index == max_rounds - 1 or (frequency > 0 and round_index % frequency == 0)


def _wandb_round_payload(record: RoundRecord) -> dict[str, Any]:
    """Flatten one round record into wandb-friendly scalar keys."""

    data = asdict(record)
    clients = data.pop("clients")
    payload = {f"round/{key}": value for key, value in data.items()}
    for client in clients:
        prefix = f"client/{client['client_id']}"
        for key, value in client.items():
            if key != "client_id":
                payload[f"{prefix}/{key}"] = value
    return payload


def run_centralized(config: dict[str, Any]) -> dict[str, float]:
    """Run centralized training over all client datasets.

    Example:
        ``run_centralized(config)`` returns test metrics such as MSE and MAE.
    """

    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir, config.get("runtime", {}).get("log_level", "INFO"))
    configure_torch_runtime(config)
    tracker = Tracker(config)
    start_time = time.perf_counter()
    device = resolve_device(config)
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    model = build_model(config).to(device)
    model_state = serialize_model(model)
    logger.info(
        "Centralized model initialized with {} parameters ({} bytes)",
        state_num_parameters(model_state),
        state_num_bytes(model_state),
    )
    tracker.log({
        "run/model_parameters": state_num_parameters(model_state),
        "run/model_bytes": state_num_bytes(model_state),
        "run/mode": "centralized",
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"].get("lr", 1e-3)))
    stopper = EarlyStopper(int(config["training"].get("patience", 5)))
    history = []
    for epoch in range(int(config["training"].get("epochs", 10))):
        epoch_start = time.perf_counter()
        loss = sum(train_one_epoch(model, loader, optimizer, device) for loader in train_loaders.values()) / len(train_loaders)
        metrics = evaluate(model, val_loader, device)
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - start_time
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss,
            "val_mse": metrics["mse"],
            "val_mae": metrics["mae"],
            "val_mape": metrics["mape"],
            "epoch_time_seconds": epoch_time,
            "elapsed_time_seconds": elapsed,
        }
        history.append(epoch_record)
        logger.info(
            "Centralized epoch {} loss={:.6f} val_mse={:.6f} time={:.2f}s elapsed={:.2f}s",
            epoch,
            loss,
            metrics["mse"],
            epoch_time,
            elapsed,
        )
        tracker.log({
            "epoch/loss": loss,
            "epoch/val_mse": metrics["mse"],
            "epoch/val_mae": metrics["mae"],
            "epoch/val_mape": metrics["mape"],
            "epoch/time_seconds": epoch_time,
            "run/elapsed_time_seconds": elapsed,
        }, step=epoch)
        if stopper.update(metrics["mse"]):
            logger.info("Centralized early stopping at epoch {}", epoch)
            break
    torch.save(model.state_dict(), output_dir / "centralized_model.pt")
    test_metrics = evaluate(model, test_loader, device)
    total_elapsed = time.perf_counter() - start_time
    config_formats = config.get("artifacts", {}).get("config_formats")
    saved_configs = save_experiment_config(config, output_dir, config_formats)
    logger.info("Saved centralized config artifacts: {}", [str(path) for path in saved_configs])
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "test": test_metrics, "epochs": len(history), "total_time_seconds": total_elapsed}, handle, ensure_ascii=False, indent=2)
    tracker.log({**{f"test/{key}": value for key, value in test_metrics.items()}, "run/total_time_seconds": total_elapsed})
    tracker.finish()
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"test": test_metrics, "epochs": len(history), "total_time_seconds": total_elapsed}, handle, ensure_ascii=False, indent=2)
    logger.info("Centralized training finished in {:.2f}s with test metrics {}", total_elapsed, test_metrics)
    return test_metrics


def run_federated(config: dict[str, Any]) -> dict[str, Any]:
    """Run single-process federated training.

    Example:
        Set ``federated.algorithm=fedavg`` for standard dense FedAvg, or
        ``compressed_fedavg`` for sparse uploads.
    """

    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir, config.get("runtime", {}).get("log_level", "INFO"))
    configure_torch_runtime(config)
    tracker = Tracker(config)
    start_time = time.perf_counter()
    device = resolve_device(config)
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    server = FederatedServer(config, val_loader, test_loader, device)
    clients = [FederatedClient(client_id, loader, config, device) for client_id, loader in train_loaders.items()]
    compressed = is_compressed_algorithm(config)
    algorithm = str(config["federated"].get("algorithm", "fedavg"))
    logger.info(
        "Starting federated run algorithm={} clients={} compressed_uploads={}",
        algorithm,
        [client.client_id for client in clients],
        compressed,
    )
    tracker.log({
        "run/algorithm": algorithm,
        "run/client_count": len(clients),
        "run/model_parameters": state_num_parameters(server.global_state),
        "run/model_bytes": state_num_bytes(server.global_state),
        "run/compressed_uploads": compressed,
    })
    stopper = EarlyStopper(int(config["training"].get("patience", 5)), float(config["training"].get("min_delta", 0.0)))
    max_rounds = int(config["federated"].get("rounds", 20))
    attack_results = []
    for round_index in range(max_rounds):
        round_start = time.perf_counter()
        results = [client.train(server.global_state, compressed=compressed) for client in clients]
        if compressed:
            server.aggregate_sparse(results)
        else:
            server.aggregate_dense(results)
        metrics = server.evaluate_global()
        record = server.record_round(
            round_index,
            results,
            metrics,
            round_time_seconds=time.perf_counter() - round_start,
            elapsed_time_seconds=time.perf_counter() - start_time,
        )
        tracker.log(_wandb_round_payload(record), step=round_index)
        if _should_run_attack(config, round_index, max_rounds):
            attack_cfg = config.get("attack", {})
            attack_start = time.perf_counter()
            grads, real_x, real_y = clients[0].gradient_sample(
                server.global_state,
                max_samples=int(attack_cfg.get("max_samples", 1)),
            )
            round_attacks = [
                dlg_attack(config, server.global_state, grads, real_x, real_y, device),
                idlg_attack(config, server.global_state, grads, real_x, real_y, device),
            ]
            attack_results.extend(round_attacks)
            attack_payload = {
                f"attack/{result.name}/reconstruction_mse": result.reconstruction_mse for result in round_attacks
            }
            attack_payload.update({
                f"attack/{result.name}/success": float(result.success) for result in round_attacks
            })
            attack_payload["attack/time_seconds"] = time.perf_counter() - attack_start
            attack_payload["attack/success_rate_so_far"] = attack_success_rate(attack_results)
            tracker.log(attack_payload, step=round_index)
            logger.info("Round {} attack metrics {}", round_index, attack_payload)
        if stopper.update(metrics["mse"]):
            logger.info("Early stopping at round {}", round_index)
            break
    test_metrics = server.test_global()
    total_elapsed = time.perf_counter() - start_time
    server.save(output_dir, config)
    tracker.log({**{f"test/{key}": value for key, value in test_metrics.items()}, "run/total_time_seconds": total_elapsed})
    tracker.finish()
    attack_records = [
        {"name": result.name, "reconstruction_mse": result.reconstruction_mse, "success": result.success}
        for result in attack_results
    ]
    with (output_dir / "attack_results.json").open("w", encoding="utf-8") as handle:
        json.dump(attack_records, handle, ensure_ascii=False, indent=2)
    summary = {
        "test": test_metrics,
        "rounds": len(server.history),
        "total_time_seconds": total_elapsed,
        "last_upload_compression_ratio": server.history[-1].upload_compression_ratio if server.history else 0.0,
        "last_total_communication_ratio": server.history[-1].total_communication_ratio if server.history else 0.0,
        "last_communication_ratio": server.history[-1].communication_ratio if server.history else 0.0,
        "attack_success_rate": attack_success_rate(attack_results),
        "attack_evaluations": len(attack_records),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("Finished experiment: {}", summary)
    return summary
