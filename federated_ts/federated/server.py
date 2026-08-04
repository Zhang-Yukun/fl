"""Federated server orchestration, aggregation, and artifact saving."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from federated_ts.utils.artifacts import save_experiment_config
from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.aggregation import fedaware_weights
from federated_ts.utils.serialization import (
    StateDict,
    add_update,
    average_states,
    decompress_topk,
    dequantize_state_update,
    quantize_state_update,
    serialize_model,
    state_num_bytes,
    state_num_parameters,
    subtract_state,
)
from federated_ts.engine.training import evaluate


@dataclass
class ClientCommunicationRecord:
    """Per-client communication and payload metadata for one FL round."""

    client_id: str
    num_samples: int
    loss: float
    payload_kind: str
    download_bytes: int
    download_parameters: int
    dense_download_reference_bytes: int
    dense_download_reference_parameters: int
    upload_bytes: int
    upload_parameters: int
    dense_upload_reference_bytes: int
    dense_upload_reference_parameters: int
    compressor: str
    privacy_clip_norm: float
    privacy_noise_multiplier: float
    aggregation_weight: float


@dataclass
class RoundRecord:
    """Metrics and communication metadata for one federated round.

    Example:
        Records are serialized into ``metrics.json`` after training.
    """

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
    """Validation-loss early stopping helper.

    Example:
        ``stopper.update(val_mse)`` returns ``True`` when patience is exhausted.
    """

    patience: int
    min_delta: float = 0.0
    best: float = float("inf")
    bad_rounds: int = 0

    def update(self, value: float) -> bool:
        """Update the best metric and report whether training should stop."""

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
        """Build the initial global model after dataclass initialization."""

        self.model = build_model(self.config).to(self.device)
        self.global_state = serialize_model(self.model)
        logger.info(
            "Initialized global model with {} parameters ({} bytes)",
            state_num_parameters(self.global_state),
            state_num_bytes(self.global_state),
        )

    def aggregate_dense(self, results) -> list[float]:
        """Aggregate dense client payloads for FedAvg-style methods."""

        algorithm = str(self.config.get("federated", {}).get("algorithm", "fedavg")).lower()
        sample_weights = [result.num_samples for result in results]
        if algorithm == "fedaware":
            aware_cfg = self.config.get("fedaware", {})
            updates = [result.state for result in results]
            weights = fedaware_weights(
                updates,
                sample_weights,
                alpha=float(aware_cfg.get("alpha", 0.5)),
                steps=int(aware_cfg.get("steps", 50)),
                lr=float(aware_cfg.get("lr", 0.1)),
            )
            averaged_update = None
            for update, weight in zip(updates, weights):
                scaled = OrderedDict((name, tensor * weight) for name, tensor in update.items())
                averaged_update = scaled if averaged_update is None else OrderedDict(
                    (name, averaged_update[name] + scaled[name]) for name in scaled
                )
            self.global_state = add_update(self.global_state, averaged_update)
            return weights
        if algorithm == "fedpetuning":
            weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
            updated_state = OrderedDict((name, tensor.clone()) for name, tensor in self.global_state.items())
            for name in results[0].state.keys():
                updated_state[name] = sum(result.state[name] * weight for result, weight in zip(results, weights))
            self.global_state = updated_state
            return weights
        if algorithm == "secure_quantized_fedavg":
            weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
            dense_updates = [dequantize_state_update(result.state) for result in results]
            averaged_update = average_states(dense_updates, sample_weights)
            quantization_dtype = str(self.config.get("federated", {}).get("quantization_dtype", "float16"))
            compressed_base = dequantize_state_update(quantize_state_update(self.global_state, dtype=quantization_dtype))
            self.global_state = add_update(compressed_base, averaged_update)
            return weights
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        averaged_update = average_states([result.state for result in results], sample_weights)
        self.global_state = add_update(self.global_state, averaged_update)
        return weights

    def aggregate_sparse(self, results) -> list[float]:
        """Aggregate sparse client updates for compressed FedAvg."""

        weights = [result.num_samples for result in results]
        total = float(sum(weights))
        update = None
        for result, weight in zip(results, weights):
            dense = decompress_topk(result.sparse_update)
            scaled = {name: tensor * (weight / total) for name, tensor in dense.items()}
            update = scaled if update is None else {name: update[name] + scaled[name] for name in scaled}
        self.global_state = add_update(self.global_state, update)
        return [weight / total for weight in weights]

    def evaluate_global(self) -> dict[str, float]:
        """Evaluate the current global model on the server validation set."""

        self.model.load_state_dict(self.global_state)
        return evaluate(self.model, self.val_loader, self.device)

    def test_global(self) -> dict[str, float]:
        """Evaluate the current global model on the server test set."""

        self.model.load_state_dict(self.global_state)
        return evaluate(self.model, self.test_loader, self.device)

    def record_round(
        self,
        round_index: int,
        results,
        aggregation_weights: list[float],
        metrics: dict[str, float],
        round_time_seconds: float,
        elapsed_time_seconds: float,
    ) -> RoundRecord:
        """Create and log a round record with communication metadata.

        Example:
            Called once after client aggregation and validation evaluation.
        """

        model_bytes = state_num_bytes(self.global_state)
        model_parameters = state_num_parameters(self.global_state)
        total_download_bytes = sum(result.download_bytes for result in results)
        total_download_parameters = sum(result.download_parameters for result in results)
        total_upload_bytes = sum(result.upload_bytes for result in results)
        total_upload_parameters = sum(result.upload_parameters for result in results)
        fedavg_reference_download_bytes = sum(result.dense_download_reference_bytes for result in results)
        fedavg_reference_upload_bytes = sum(result.dense_bytes for result in results)
        fedavg_reference_upload_parameters = sum(result.dense_parameters for result in results)
        fedavg_reference_total_bytes = fedavg_reference_download_bytes + fedavg_reference_upload_bytes
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
                dense_download_reference_bytes=result.dense_download_reference_bytes,
                dense_download_reference_parameters=result.dense_download_reference_parameters,
                upload_bytes=result.upload_bytes,
                upload_parameters=result.upload_parameters,
                dense_upload_reference_bytes=result.dense_bytes,
                dense_upload_reference_parameters=result.dense_parameters,
                compressor=result.compressor,
                privacy_clip_norm=result.privacy_clip_norm,
                privacy_noise_multiplier=result.privacy_noise_multiplier,
                aggregation_weight=aggregation_weight,
            )
            for result, aggregation_weight in zip(results, aggregation_weights)
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
                "Round {} client={} samples={} payload={} compressor={} clip_norm={} noise_multiplier={} agg_weight={:.6f} loss={:.6f} upload={}B/{} params download={}B/{} params",
                round_index,
                client.client_id,
                client.num_samples,
                client.payload_kind,
                client.compressor,
                client.privacy_clip_norm,
                client.privacy_noise_multiplier,
                client.aggregation_weight,
                client.loss,
                client.upload_bytes,
                client.upload_parameters,
                client.download_bytes,
                client.download_parameters,
            )
        return record

    def save(self, output_dir: str | Path, config: dict[str, Any]) -> None:
        """Persist model, experiment parameters, and metric history.

        Example:
            ``server.save("outputs/run", config)`` writes ``model.pt``,
            ``metrics.json``, and parameter files such as ``config.yaml``.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_state, output_dir / "model.pt")
        config_formats = config.get("artifacts", {}).get("config_formats")
        saved_configs = save_experiment_config(config, output_dir, config_formats)
        logger.info("Saved experiment config artifacts: {}", [str(path) for path in saved_configs])
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump([asdict(record) for record in self.history], handle, ensure_ascii=False, indent=2)
