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
)
from federated_ts.engine.training import evaluate
from federated_ts.utils.privacy_accounting import (
    AdaptiveClippedRdpAccountant,
    AdaptiveRdpStep,
    adaptive_clip_threshold,
    add_gaussian_noise,
    clip_state_to_norm,
    state_l2_norm,
)


def _format_num_bytes(num_bytes: int) -> str:
    """Format raw bytes into a compact human-readable string."""

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


@dataclass
class ClientCommunicationRecord:
    """Per-client communication and payload metadata for one FL round."""

    client_id: str
    num_samples: int
    loss: float
    payload_kind: str
    download_bytes: int
    download_parameters: int
    parameter_download_bytes: int
    parameter_download_parameters: int
    dense_download_reference_bytes: int
    dense_download_reference_parameters: int
    upload_bytes: int
    upload_parameters: int
    parameter_upload_bytes: int
    parameter_upload_parameters: int
    transport_download_bytes: int
    transport_upload_bytes: int
    transport_download_overhead_bytes: int
    transport_upload_overhead_bytes: int
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
    total_parameter_download_bytes: int
    total_parameter_download_parameters: int
    total_parameter_upload_bytes: int
    total_parameter_upload_parameters: int
    total_parameter_bytes: int
    total_transport_download_bytes: int
    total_transport_upload_bytes: int
    total_transport_bytes: int
    total_transport_download_overhead_bytes: int
    total_transport_upload_overhead_bytes: int
    fedavg_reference_upload_bytes: int
    fedavg_reference_upload_parameters: int
    fedavg_reference_total_bytes: int
    upload_compression_ratio: float
    total_communication_ratio: float
    communication_ratio: float
    transport_upload_compression_ratio: float
    transport_total_communication_ratio: float
    privacy_accountant: str | None = None
    privacy_epsilon: float | None = None
    privacy_delta: float | None = None
    privacy_rdp_alpha: float | None = None
    privacy_rdp_round: float | None = None
    privacy_rdp_total: float | None = None
    privacy_sampling_rate: float | None = None
    adaptive_clip_norm: float | None = None
    adaptive_clip_raw: float | None = None
    adaptive_clip_median_norm: float | None = None
    adaptive_reference_clip_norm: float | None = None
    adaptive_noise_std: float | None = None
    clients: list[ClientCommunicationRecord] = field(default_factory=list)


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
    adaptive_accountant: AdaptiveClippedRdpAccountant | None = field(init=False, default=None)
    last_privacy_step: AdaptiveRdpStep | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Build the initial global model after dataclass initialization."""

        self.model = build_model(self.config).to(self.device)
        self.global_state = serialize_model(self.model)
        adaptive_cfg = self.config.get("adaptive_clipped_rdp", {})
        if str(self.config.get("federated", {}).get("algorithm", "fedavg")).lower() == "adaptive_clipped_rdp_fedavg":
            self.adaptive_accountant = AdaptiveClippedRdpAccountant(
                rdp_alpha=float(adaptive_cfg.get("rdp_alpha", 16.0)),
                delta=float(adaptive_cfg.get("delta", 1e-5)),
                noise_multiplier=float(adaptive_cfg.get("noise_multiplier", 0.0)),
            )
        logger.info(
            "Initialized global model with {} parameters ({} bytes)",
            state_num_parameters(self.global_state),
            state_num_bytes(self.global_state),
        )

    def _adaptive_noise_generator(self, round_index: int) -> torch.Generator | None:
        """Create a deterministic server-side generator for adaptive DP noise."""

        seed = self.config.get("adaptive_clipped_rdp", {}).get("seed")
        if seed is None:
            seed = self.config.get("runtime", {}).get("seed")
        if seed is None:
            return None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(round_index) * 1009)
        return generator

    def _aggregate_adaptive_clipped_rdp(self, results, round_index: int) -> list[float]:
        """Aggregate raw client updates with adaptive server-side clipping and RDP accounting."""

        adaptive_cfg = self.config.get("adaptive_clipped_rdp", {})
        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        raw_updates = [result.state for result in results]
        update_norms = [state_l2_norm(update) for update in raw_updates]
        median_norm, raw_clip, clip_norm = adaptive_clip_threshold(
            update_norms,
            clip_factor=float(adaptive_cfg.get("clip_factor", 1.0)),
            min_clip=float(adaptive_cfg.get("min_clip_norm", 0.0)),
            max_clip=float(adaptive_cfg.get("max_clip_norm", float(adaptive_cfg.get("reference_clip_norm", 1.0)))),
        )
        clipped_updates = [clip_state_to_norm(update, clip_norm)[0] for update in raw_updates]
        averaged_update = average_states(clipped_updates, sample_weights)
        reference_clip_norm = float(adaptive_cfg.get("reference_clip_norm", clip_norm))
        noise_multiplier = float(adaptive_cfg.get("noise_multiplier", 0.0))
        noise_std = noise_multiplier * reference_clip_norm
        noisy_update = add_gaussian_noise(
            averaged_update,
            noise_std=noise_std,
            generator=self._adaptive_noise_generator(round_index),
        )
        self.global_state = add_update(self.global_state, noisy_update)
        total_clients = int(adaptive_cfg.get("total_clients", len(results)))
        sampling_rate = float(len(results)) / float(max(total_clients, 1))
        if self.adaptive_accountant is not None:
            self.last_privacy_step = self.adaptive_accountant.step(
                round_index=round_index,
                sampling_rate=sampling_rate,
                adaptive_clip_norm=clip_norm,
                reference_clip_norm=reference_clip_norm,
                median_update_norm=median_norm,
                raw_clip_norm=raw_clip,
            )
        return weights

    def aggregate_dense(self, results, round_index: int = 0) -> list[float]:
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
        if algorithm == "secure_quantized_fedavg":
            weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
            dense_updates = [dequantize_state_update(result.state) for result in results]
            averaged_update = average_states(dense_updates, sample_weights)
            quantization_dtype = str(self.config.get("federated", {}).get("quantization_dtype", "float16"))
            compressed_base = dequantize_state_update(quantize_state_update(self.global_state, dtype=quantization_dtype))
            self.global_state = add_update(compressed_base, averaged_update)
            return weights
        if algorithm == "adaptive_clipped_rdp_fedavg":
            return self._aggregate_adaptive_clipped_rdp(results, round_index)
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
        total_parameter_download_bytes = sum(result.parameter_download_bytes for result in results)
        total_parameter_download_parameters = sum(result.parameter_download_parameters for result in results)
        total_parameter_upload_bytes = sum(result.parameter_upload_bytes for result in results)
        total_parameter_upload_parameters = sum(result.parameter_upload_parameters for result in results)
        total_transport_download_bytes = sum(result.transport_download_bytes for result in results)
        total_transport_upload_bytes = sum(result.transport_upload_bytes for result in results)
        total_transport_download_overhead_bytes = sum(result.transport_download_overhead_bytes for result in results)
        total_transport_upload_overhead_bytes = sum(result.transport_upload_overhead_bytes for result in results)
        total_download_bytes = total_parameter_download_bytes
        total_download_parameters = total_parameter_download_parameters
        total_upload_bytes = total_parameter_upload_bytes
        total_upload_parameters = total_parameter_upload_parameters
        fedavg_reference_download_bytes = sum(result.dense_download_reference_bytes for result in results)
        fedavg_reference_upload_bytes = sum(result.dense_bytes for result in results)
        fedavg_reference_upload_parameters = sum(result.dense_parameters for result in results)
        fedavg_reference_total_bytes = fedavg_reference_download_bytes + fedavg_reference_upload_bytes
        total_parameter_bytes = total_parameter_download_bytes + total_parameter_upload_bytes
        total_transport_bytes = total_transport_download_bytes + total_transport_upload_bytes
        upload_ratio = fedavg_reference_upload_bytes / max(total_parameter_upload_bytes, 1)
        total_ratio = fedavg_reference_total_bytes / max(total_parameter_bytes, 1)
        transport_upload_ratio = fedavg_reference_upload_bytes / max(total_transport_upload_bytes, 1)
        transport_total_ratio = fedavg_reference_total_bytes / max(total_transport_bytes, 1)
        client_records = [
            ClientCommunicationRecord(
                client_id=result.client_id,
                num_samples=result.num_samples,
                loss=result.loss,
                payload_kind=result.payload_kind,
                download_bytes=result.parameter_download_bytes,
                download_parameters=result.parameter_download_parameters,
                parameter_download_bytes=result.parameter_download_bytes,
                parameter_download_parameters=result.parameter_download_parameters,
                dense_download_reference_bytes=result.dense_download_reference_bytes,
                dense_download_reference_parameters=result.dense_download_reference_parameters,
                upload_bytes=result.parameter_upload_bytes,
                upload_parameters=result.parameter_upload_parameters,
                parameter_upload_bytes=result.parameter_upload_bytes,
                parameter_upload_parameters=result.parameter_upload_parameters,
                transport_download_bytes=result.transport_download_bytes,
                transport_upload_bytes=result.transport_upload_bytes,
                transport_download_overhead_bytes=result.transport_download_overhead_bytes,
                transport_upload_overhead_bytes=result.transport_upload_overhead_bytes,
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
            total_parameter_download_bytes=total_parameter_download_bytes,
            total_parameter_download_parameters=total_parameter_download_parameters,
            total_parameter_upload_bytes=total_parameter_upload_bytes,
            total_parameter_upload_parameters=total_parameter_upload_parameters,
            total_parameter_bytes=total_parameter_bytes,
            total_transport_download_bytes=total_transport_download_bytes,
            total_transport_upload_bytes=total_transport_upload_bytes,
            total_transport_bytes=total_transport_bytes,
            total_transport_download_overhead_bytes=total_transport_download_overhead_bytes,
            total_transport_upload_overhead_bytes=total_transport_upload_overhead_bytes,
            fedavg_reference_upload_bytes=fedavg_reference_upload_bytes,
            fedavg_reference_upload_parameters=fedavg_reference_upload_parameters,
            fedavg_reference_total_bytes=fedavg_reference_total_bytes,
            upload_compression_ratio=upload_ratio,
            total_communication_ratio=total_ratio,
            communication_ratio=upload_ratio,
            transport_upload_compression_ratio=transport_upload_ratio,
            transport_total_communication_ratio=transport_total_ratio,
            privacy_accountant=None if self.last_privacy_step is None else "adaptive_clipped_rdp",
            privacy_epsilon=None if self.last_privacy_step is None else self.last_privacy_step.epsilon,
            privacy_delta=None if self.last_privacy_step is None else self.last_privacy_step.delta,
            privacy_rdp_alpha=None if self.last_privacy_step is None else self.last_privacy_step.rdp_alpha,
            privacy_rdp_round=None if self.last_privacy_step is None else self.last_privacy_step.round_rdp,
            privacy_rdp_total=None if self.last_privacy_step is None else self.last_privacy_step.total_rdp,
            privacy_sampling_rate=None if self.last_privacy_step is None else self.last_privacy_step.sampling_rate,
            adaptive_clip_norm=None if self.last_privacy_step is None else self.last_privacy_step.adaptive_clip_norm,
            adaptive_clip_raw=None if self.last_privacy_step is None else self.last_privacy_step.raw_clip_norm,
            adaptive_clip_median_norm=None if self.last_privacy_step is None else self.last_privacy_step.median_update_norm,
            adaptive_reference_clip_norm=None if self.last_privacy_step is None else self.last_privacy_step.reference_clip_norm,
            adaptive_noise_std=None if self.last_privacy_step is None else self.last_privacy_step.noise_std,
            clients=client_records,
        )
        self.history.append(record)
        cumulative_parameter_upload_bytes = sum(item.total_parameter_upload_bytes for item in self.history)
        cumulative_parameter_download_bytes = sum(item.total_parameter_download_bytes for item in self.history)
        cumulative_transport_upload_bytes = sum(item.total_transport_upload_bytes for item in self.history)
        cumulative_transport_download_bytes = sum(item.total_transport_download_bytes for item in self.history)
        logger.info(
            "Round {} algorithm={} val_mse={:.6f} time={:.2f}s parameter_upload={} ({}) parameter_download={} ({}) transport_upload={} ({}) transport_download={} ({}) upload_ratio={:.2f} total_ratio={:.2f}",
            round_index,
            record.algorithm,
            record.val_mse,
            record.round_time_seconds,
            record.total_upload_bytes,
            _format_num_bytes(record.total_upload_bytes),
            record.total_download_bytes,
            _format_num_bytes(record.total_download_bytes),
            record.total_transport_upload_bytes,
            _format_num_bytes(record.total_transport_upload_bytes),
            record.total_transport_download_bytes,
            _format_num_bytes(record.total_transport_download_bytes),
            record.upload_compression_ratio,
            record.total_communication_ratio,
        )
        if self.last_privacy_step is not None:
            logger.info(
                "Round {} adaptive_rdp clip_median={:.6f} raw_clip={:.6f} clip_norm={:.6f} noise_std={:.6f} q={:.6f} rdp_round={:.6f} rdp_total={:.6f} epsilon={:.6f} delta={:.2e}",
                round_index,
                self.last_privacy_step.median_update_norm,
                self.last_privacy_step.raw_clip_norm,
                self.last_privacy_step.adaptive_clip_norm,
                self.last_privacy_step.noise_std,
                self.last_privacy_step.sampling_rate,
                self.last_privacy_step.round_rdp,
                self.last_privacy_step.total_rdp,
                self.last_privacy_step.epsilon,
                self.last_privacy_step.delta,
            )
        logger.info(
            "Round {} cumulative parameter_upload={} ({}) parameter_download={} ({}) transport_upload={} ({}) transport_download={} ({})",
            round_index,
            cumulative_parameter_upload_bytes,
            _format_num_bytes(cumulative_parameter_upload_bytes),
            cumulative_parameter_download_bytes,
            _format_num_bytes(cumulative_parameter_download_bytes),
            cumulative_transport_upload_bytes,
            _format_num_bytes(cumulative_transport_upload_bytes),
            cumulative_transport_download_bytes,
            _format_num_bytes(cumulative_transport_download_bytes),
        )
        for client in client_records:
            logger.info(
                "Round {} client={} samples={} payload={} compressor={} clip_norm={} noise_multiplier={} agg_weight={:.6f} loss={:.6f} parameter_upload={} ({})/{} params parameter_download={} ({})/{} params transport_upload={} ({}) transport_download={} ({})",
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
                _format_num_bytes(client.upload_bytes),
                client.upload_parameters,
                client.download_bytes,
                _format_num_bytes(client.download_bytes),
                client.download_parameters,
                client.transport_upload_bytes,
                _format_num_bytes(client.transport_upload_bytes),
                client.transport_download_bytes,
                _format_num_bytes(client.transport_download_bytes),
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
