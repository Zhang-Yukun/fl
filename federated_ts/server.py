"""Federated server orchestration, aggregation, and artifact saving."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from federated_ts.models import build_model
from federated_ts.serialization import (
    StateDict,
    add_update,
    average_states,
    decompress_topk,
    serialize_model,
    state_num_bytes,
    state_num_parameters,
)
from federated_ts.training import evaluate


@dataclass
class ClientCommunicationRecord:
    """Per-client communication and payload metadata for one FL round."""

    client_id: str
    num_samples: int
    loss: float
    payload_kind: str
    download_bytes: int
    download_parameters: int
    upload_bytes: int
    upload_parameters: int
    dense_upload_reference_bytes: int
    dense_upload_reference_parameters: int


@dataclass
class RoundRecord:
    round: int
    algorithm: str
    train_loss: float
    val_mse: float
    val_mae: float
    val_mape: float
    round_time_seconds: float
    elapsed_time_seconds: float
    model_parameters: int
    model_bytes: int
    total_download_bytes: int
    total_download_parameters: int
    total_upload_bytes: int
    total_upload_parameters: int
    fedavg_reference_upload_bytes: int
    fedavg_reference_upload_parameters: int
    fedavg_reference_total_bytes: int
    upload_compression_ratio: float
    total_communication_ratio: float
    communication_ratio: float
    clients: list[ClientCommunicationRecord]


@dataclass
class EarlyStopper:
    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    bad_rounds: int = 0

    def update(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_rounds = 0
            return False
        self.bad_rounds += 1
        return self.bad_rounds >= self.patience


@dataclass
class FederatedServer:
    """Aggregates client results and owns server-side validation/test data."""

    config: dict[str, Any]
    val_loader: Any
    test_loader: Any
    device: torch.device
    model: torch.nn.Module = field(init=False)
    global_state: StateDict = field(init=False)
    history: list[RoundRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model = build_model(self.config).to(self.device)
        self.global_state = serialize_model(self.model)
        logger.info(
            "Initialized global model with {} parameters ({} bytes)",
            state_num_parameters(self.global_state),
            state_num_bytes(self.global_state),
        )

    def aggregate_dense(self, results) -> None:
        weights = [result.num_samples for result in results]
        self.global_state = average_states([result.state for result in results], weights)

    def aggregate_sparse(self, results) -> None:
        weights = [result.num_samples for result in results]
        total = float(sum(weights))
        update = None
        for result, weight in zip(results, weights):
            dense = decompress_topk(result.sparse_update)
            scaled = {name: tensor * (weight / total) for name, tensor in dense.items()}
            update = scaled if update is None else {name: update[name] + scaled[name] for name in scaled}
        self.global_state = add_update(self.global_state, update)

    def evaluate_global(self) -> dict[str, float]:
        self.model.load_state_dict(self.global_state)
        return evaluate(self.model, self.val_loader, self.device)

    def test_global(self) -> dict[str, float]:
        self.model.load_state_dict(self.global_state)
        return evaluate(self.model, self.test_loader, self.device)

    def record_round(
        self,
        round_index: int,
        results,
        metrics: dict[str, float],
        round_time_seconds: float,
        elapsed_time_seconds: float,
    ) -> RoundRecord:
        model_bytes = state_num_bytes(self.global_state)
        model_parameters = state_num_parameters(self.global_state)
        total_download_bytes = sum(result.download_bytes for result in results)
        total_download_parameters = sum(result.download_parameters for result in results)
        total_upload_bytes = sum(result.upload_bytes for result in results)
        total_upload_parameters = sum(result.upload_parameters for result in results)
        fedavg_reference_upload_bytes = sum(result.dense_bytes for result in results)
        fedavg_reference_upload_parameters = sum(result.dense_parameters for result in results)
        fedavg_reference_total_bytes = total_download_bytes + fedavg_reference_upload_bytes
        upload_ratio = fedavg_reference_upload_bytes / max(total_upload_bytes, 1)
        total_ratio = fedavg_reference_total_bytes / max(total_download_bytes + total_upload_bytes, 1)
        client_records = [
            ClientCommunicationRecord(
                client_id=result.client_id,
                num_samples=result.num_samples,
                loss=result.loss,
                payload_kind=result.payload_kind,
                download_bytes=result.download_bytes,
                download_parameters=result.download_parameters,
                upload_bytes=result.upload_bytes,
                upload_parameters=result.upload_parameters,
                dense_upload_reference_bytes=result.dense_bytes,
                dense_upload_reference_parameters=result.dense_parameters,
            )
            for result in results
        ]
        record = RoundRecord(
            round=round_index,
            algorithm=str(self.config.get("federated", {}).get("algorithm", "fedavg")),
            train_loss=sum(r.loss for r in results) / len(results),
            val_mse=metrics["mse"],
            val_mae=metrics["mae"],
            val_mape=metrics["mape"],
            round_time_seconds=round_time_seconds,
            elapsed_time_seconds=elapsed_time_seconds,
            model_parameters=model_parameters,
            model_bytes=model_bytes,
            total_download_bytes=total_download_bytes,
            total_download_parameters=total_download_parameters,
            total_upload_bytes=total_upload_bytes,
            total_upload_parameters=total_upload_parameters,
            fedavg_reference_upload_bytes=fedavg_reference_upload_bytes,
            fedavg_reference_upload_parameters=fedavg_reference_upload_parameters,
            fedavg_reference_total_bytes=fedavg_reference_total_bytes,
            upload_compression_ratio=upload_ratio,
            total_communication_ratio=total_ratio,
            communication_ratio=upload_ratio,
            clients=client_records,
        )
        self.history.append(record)
        logger.info(
            "Round {} algorithm={} val_mse={:.6f} time={:.2f}s upload={}B download={}B upload_ratio={:.2f} total_ratio={:.2f}",
            round_index,
            record.algorithm,
            record.val_mse,
            record.round_time_seconds,
            record.total_upload_bytes,
            record.total_download_bytes,
            record.upload_compression_ratio,
            record.total_communication_ratio,
        )
        for client in client_records:
            logger.info(
                "Round {} client={} samples={} payload={} loss={:.6f} upload={}B/{} params download={}B/{} params",
                round_index,
                client.client_id,
                client.num_samples,
                client.payload_kind,
                client.loss,
                client.upload_bytes,
                client.upload_parameters,
                client.download_bytes,
                client.download_parameters,
            )
        return record

    def save(self, output_dir: str | Path, config: dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_state, output_dir / "model.pt")
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump([asdict(record) for record in self.history], handle, ensure_ascii=False, indent=2)
