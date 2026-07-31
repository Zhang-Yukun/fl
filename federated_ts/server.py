"""Federated server orchestration, aggregation, and artifact saving."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from federated_ts.models import build_model
from federated_ts.serialization import StateDict, add_update, average_states, decompress_topk, serialize_model, state_num_bytes
from federated_ts.training import evaluate


@dataclass
class RoundRecord:
    round: int
    train_loss: float
    val_mse: float
    val_mae: float
    val_mape: float
    communication_ratio: float


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

    def record_round(self, round_index: int, results, metrics: dict[str, float]) -> RoundRecord:
        fedavg_bytes = state_num_bytes(self.global_state) * len(results)
        sent_bytes = sum(result.sent_bytes for result in results)
        ratio = fedavg_bytes / max(sent_bytes, 1)
        record = RoundRecord(
            round=round_index,
            train_loss=sum(r.loss for r in results) / len(results),
            val_mse=metrics["mse"],
            val_mae=metrics["mae"],
            val_mape=metrics["mape"],
            communication_ratio=ratio,
        )
        self.history.append(record)
        logger.info("Round {} val_mse={:.6f} comm_ratio={:.2f}", round_index, record.val_mse, record.communication_ratio)
        return record

    def save(self, output_dir: str | Path, config: dict[str, Any]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_state, output_dir / "model.pt")
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump([record.__dict__ for record in self.history], handle, ensure_ascii=False, indent=2)

