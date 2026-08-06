"""High-level training algorithms: centralized, FedAvg, and compressed FedAvg."""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in the training env
    np = None
from loguru import logger

from federated_ts.utils.artifacts import save_experiment_config
from federated_ts.security.attacks import attack_success_rate, dlg_attack, idlg_attack, summarize_attack_results
from federated_ts.federated.client import FederatedClient
from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.utils.logging import setup_logging
from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import (
    StateDict,
    decompress_topk,
    dequantize_state_update,
    serialize_model,
    state_num_bytes,
    state_num_parameters,
)
from federated_ts.federated.server import EarlyStopper, FederatedServer, RoundRecord
from federated_ts.utils.tracking import Tracker
from federated_ts.engine.training import evaluate, train_one_epoch


def configure_torch_runtime(config: dict[str, Any]) -> None:
    """Apply CPU thread limits from runtime config before training starts."""

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


def setup_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and torch random sources to a reproducible state.

    Example:
        ``setup_seed(2026, deterministic=True)`` makes repeated local runs
        reproduce the same model init and dataloader shuffles.
    """

    seed_value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed_value)
    if np is not None:
        np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception as exc:  # pragma: no cover - depends on torch build/runtime
        logger.warning("Could not set deterministic torch algorithms: {}", exc)
    logger.info("Set runtime seed={} deterministic={}", seed_value, deterministic)


def configure_random_seed(config: dict[str, Any]) -> None:
    """Apply the configured runtime seed when one is provided."""

    runtime_cfg = config.get("runtime", {})
    seed = runtime_cfg.get("seed")
    if seed is None:
        return
    setup_seed(int(seed), deterministic=bool(runtime_cfg.get("deterministic", True)))


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve the configured torch device with a CPU fallback."""

    requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    return torch.device(requested)


def is_compressed_algorithm(config: dict[str, Any]) -> bool:
    """Return whether the configured FL algorithm compresses client uploads."""

    algorithm = str(config.get("federated", {}).get("algorithm", "fedavg")).lower()
    if algorithm in {"fedavg", "fedaware", "fedpetuning", "secure_quantized_fedavg"}:
        return False
    if algorithm in {"compressed_fedavg", "sparse_fedavg", "soteriafl", "dp_topk_fedavg"}:
        return True
    raise ValueError(f"Unknown federated algorithm: {algorithm}")


def _should_run_attack(config: dict[str, Any], round_index: int, max_rounds: int) -> bool:
    """Return whether attack evaluation should run on this round."""

    attack_cfg = config.get("attack", {})
    if not attack_cfg.get("enabled", True):
        return False
    frequency = int(attack_cfg.get("frequency_rounds", 1))
    return round_index == 0 or round_index == max_rounds - 1 or (frequency > 0 and round_index % frequency == 0)


def _select_attack_clients(clients: list[FederatedClient], config: dict[str, Any], round_index: int) -> list[FederatedClient]:
    """Select which clients should be attacked on the current round."""

    attack_cfg = config.get("attack", {})
    selection = str(attack_cfg.get("client_selection", "round_robin")).lower()
    count = max(1, int(attack_cfg.get("clients_per_round", 1)))
    if selection == "all":
        return clients
    if selection == "first":
        return clients[:count]
    start = round_index % len(clients)
    return [clients[(start + offset) % len(clients)] for offset in range(min(count, len(clients)))]


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


def _round_history_communication_summary(history: list[RoundRecord]) -> dict[str, float | int]:
    """Summarize parameter and transport communication over all recorded rounds."""

    if not history:
        return {
            "last_parameter_upload_bytes": 0,
            "last_parameter_download_bytes": 0,
            "last_parameter_total_bytes": 0,
            "last_transport_upload_bytes": 0,
            "last_transport_download_bytes": 0,
            "last_transport_total_bytes": 0,
            "last_transport_upload_overhead_bytes": 0,
            "last_transport_download_overhead_bytes": 0,
            "last_transport_upload_compression_ratio": 0.0,
            "last_transport_total_communication_ratio": 0.0,
            "total_parameter_upload_bytes": 0,
            "total_parameter_download_bytes": 0,
            "total_parameter_bytes": 0,
            "total_transport_upload_bytes": 0,
            "total_transport_download_bytes": 0,
            "total_transport_bytes": 0,
            "total_transport_upload_overhead_bytes": 0,
            "total_transport_download_overhead_bytes": 0,
        }
    last = history[-1]
    return {
        "last_parameter_upload_bytes": last.total_parameter_upload_bytes,
        "last_parameter_download_bytes": last.total_parameter_download_bytes,
        "last_parameter_total_bytes": last.total_parameter_bytes,
        "last_transport_upload_bytes": last.total_transport_upload_bytes,
        "last_transport_download_bytes": last.total_transport_download_bytes,
        "last_transport_total_bytes": last.total_transport_bytes,
        "last_transport_upload_overhead_bytes": last.total_transport_upload_overhead_bytes,
        "last_transport_download_overhead_bytes": last.total_transport_download_overhead_bytes,
        "last_transport_upload_compression_ratio": last.transport_upload_compression_ratio,
        "last_transport_total_communication_ratio": last.transport_total_communication_ratio,
        "total_parameter_upload_bytes": sum(record.total_parameter_upload_bytes for record in history),
        "total_parameter_download_bytes": sum(record.total_parameter_download_bytes for record in history),
        "total_parameter_bytes": sum(record.total_parameter_bytes for record in history),
        "total_transport_upload_bytes": sum(record.total_transport_upload_bytes for record in history),
        "total_transport_download_bytes": sum(record.total_transport_download_bytes for record in history),
        "total_transport_bytes": sum(record.total_transport_bytes for record in history),
        "total_transport_upload_overhead_bytes": sum(record.total_transport_upload_overhead_bytes for record in history),
        "total_transport_download_overhead_bytes": sum(record.total_transport_download_overhead_bytes for record in history),
    }


def _wandb_cumulative_communication_payload(history: list[RoundRecord]) -> dict[str, float | int]:
    """Return cumulative communication metrics in a wandb-friendly flat namespace."""

    summary = _round_history_communication_summary(history)
    return {f"cumulative/{key}": value for key, value in summary.items()}


def _protect_attack_gradients(
    config: dict[str, Any],
    grads: list[torch.Tensor],
    round_index: int,
    client_index: int,
    sample_index: int,
) -> list[torch.Tensor]:
    """Apply the configured upload protection to intercepted attack gradients.

    Example:
        ``protected = _protect_attack_gradients(config, grads, 0, 0, 0)`` mirrors
        the client-side clipping, noise, and sparsification seen by the server.
    """

    protected = [grad.detach().cpu().clone() for grad in grads]
    algorithm = str(config.get("federated", {}).get("algorithm", "fedavg")).lower()
    if not protected or algorithm in {"fedavg", "fedaware", "fedpetuning"}:
        return protected

    shapes = [tuple(grad.shape) for grad in protected]
    flat = torch.cat([grad.reshape(-1) for grad in protected])
    attack_cfg = config.get("attack", {})
    seed = attack_cfg.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + round_index * 1000 + client_index * 100 + sample_index)

    if algorithm in {"soteriafl", "dp_topk_fedavg"}:
        privacy_cfg = config.get("privacy", {})
        clip_norm = float(privacy_cfg.get("clip_norm", 1.0))
        noise_multiplier = float(privacy_cfg.get("noise_multiplier", 0.1))
        if clip_norm > 0:
            norm = torch.linalg.vector_norm(flat)
            scale = min(1.0, float(clip_norm / (norm + 1e-12)))
            flat = flat * scale
        if noise_multiplier > 0 and clip_norm > 0:
            flat = flat + torch.randn(flat.shape, generator=generator, dtype=flat.dtype) * (noise_multiplier * clip_norm)

    if algorithm == "secure_quantized_fedavg":
        privacy_cfg = config.get("privacy", {})
        clip_norm = float(privacy_cfg.get("clip_norm", 0.0))
        noise_multiplier = float(privacy_cfg.get("noise_multiplier", 0.0))
        if clip_norm > 0:
            norm = torch.linalg.vector_norm(flat)
            scale = min(1.0, float(clip_norm / (norm + 1e-12)))
            flat = flat * scale
        if noise_multiplier > 0 and clip_norm > 0:
            flat = flat + torch.randn(flat.shape, generator=generator, dtype=flat.dtype) * (noise_multiplier * clip_norm)
        quant_dtype = str(config.get("federated", {}).get("quantization_dtype", "float16")).lower()
        if quant_dtype == "float16":
            flat = flat.to(torch.float16).to(torch.float32)
        elif quant_dtype == "bfloat16":
            flat = flat.to(torch.bfloat16).to(torch.float32)
        elif quant_dtype in {"int8", "qint8", "absmax_int8", "scaled_int8"}:
            max_abs = float(flat.abs().max().item())
            scale = max(max_abs / 127.0, 1e-12)
            normalized = torch.clamp(flat / scale, -127.0, 127.0)
            if bool(config.get("federated", {}).get("quantization_stochastic_rounding", False)):
                lower = torch.floor(normalized)
                probability = normalized - lower
                random = torch.rand(normalized.shape, generator=generator, dtype=torch.float32)
                rounded = lower + (random < probability).to(torch.float32)
            else:
                rounded = torch.round(normalized)
            flat = torch.clamp(rounded, -127.0, 127.0).to(torch.int8).to(torch.float32) * scale

    if algorithm in {"compressed_fedavg", "sparse_fedavg", "dp_topk_fedavg", "soteriafl"}:
        fraction = float(config.get("federated", {}).get("topk_fraction", 0.05))
        total = flat.numel()
        k = max(1, int(total * fraction))
        sparse = torch.zeros_like(flat)
        if algorithm == "soteriafl":
            indices = torch.randperm(total, generator=generator)[:k]
            sparse[indices] = flat[indices] * (float(total) / float(k))
        else:
            _, indices = torch.topk(flat.abs(), k)
            sparse[indices] = flat[indices]
        flat = sparse

    rebuilt: list[torch.Tensor] = []
    offset = 0
    for grad, shape in zip(protected, shapes):
        length = grad.numel()
        rebuilt.append(flat[offset : offset + length].reshape(shape).to(dtype=grad.dtype))
        offset += length
    return rebuilt


def _clone_state(state: StateDict) -> StateDict:
    """Return a detached CPU clone of a serialized model state."""

    return type(state)((name, tensor.detach().cpu().clone()) for name, tensor in state.items())


def _attack_target_type(config: dict[str, Any]) -> str:
    """Return the configured interception target for reconstruction attacks."""

    return str(config.get("attack", {}).get("target_type", "update_payload")).lower()


def _extract_attack_payload(config: dict[str, Any], result) -> StateDict:
    """Return the actual transmitted client payload as a dense state update."""

    if result.sparse_update is not None:
        return decompress_topk(result.sparse_update)
    if result.state is None:
        raise ValueError(f"Client {result.client_id} did not produce an attackable payload")
    algorithm = str(config.get("federated", {}).get("algorithm", "fedavg")).lower()
    if algorithm == "secure_quantized_fedavg":
        return dequantize_state_update(result.state)
    return _clone_state(result.state)


@dataclass
class AttackSampleTask:
    """Immutable attack inputs captured from one round before async execution."""

    client_id: str
    round_index: int
    sample_index: int
    target_type: str
    round_base_state: StateDict
    target: list[torch.Tensor] | StateDict
    real_x: torch.Tensor
    real_y: torch.Tensor
    reference_inputs: torch.Tensor | None = None


@dataclass
class AttackRoundTask:
    """One round of attack work detached from the training hot path."""

    round_index: int
    clients_this_round: int
    samples_per_client: int
    samples: list[AttackSampleTask]


@dataclass
class AttackRoundResult:
    """Completed attack artifacts for one training round."""

    round_index: int
    time_seconds: float
    clients_this_round: int
    samples_per_client: int
    attacks: list[Any]


def _clone_attack_target(target: list[torch.Tensor] | StateDict) -> list[torch.Tensor] | StateDict:
    """Detach and clone an intercepted attack target onto CPU memory."""

    if isinstance(target, list):
        return [tensor.detach().cpu().clone() for tensor in target]
    return _clone_state(target)


def _resolve_attack_device(config: dict[str, Any]) -> torch.device:
    """Resolve the device used for asynchronous attack evaluation."""

    attack_cfg = config.get("attack", {})
    requested = str(attack_cfg.get("async_device", "cpu")).lower()
    if requested == "same":
        requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Async attack device {} unavailable; falling back to CPU", requested)
        requested = "cpu"
    return torch.device(requested)


def _build_attack_round_task(
    config: dict[str, Any],
    clients: list[FederatedClient],
    results,
    round_index: int,
    max_rounds: int,
    round_base_state: StateDict,
    attack_target_type: str,
) -> AttackRoundTask | None:
    """Capture one round of attack inputs as immutable CPU snapshots."""

    if not _should_run_attack(config, round_index, max_rounds):
        return None
    attack_cfg = config.get("attack", {})
    max_samples = int(attack_cfg.get("max_samples", 1))
    sample_count = max(1, int(attack_cfg.get("sample_count", 1)))
    selected_clients = _select_attack_clients(clients, config, round_index)
    results_by_client = {result.client_id: result for result in results}
    samples: list[AttackSampleTask] = []
    for client_index, client in enumerate(selected_clients):
        result = results_by_client[client.client_id]
        for sample_index in range(sample_count):
            reference_inputs = None
            if attack_target_type == "gradient":
                grads, real_x, real_y = client.gradient_sample(
                    round_base_state,
                    max_samples=max_samples,
                    batch_index=sample_index,
                )
                target = _protect_attack_gradients(
                    config,
                    grads,
                    round_index=round_index,
                    client_index=client_index,
                    sample_index=sample_index,
                )
            else:
                real_x, real_y = client.sample_batch(max_samples=max_samples, batch_index=sample_index)
                target = _extract_attack_payload(config, result)
                reference_inputs = client.train_reference_inputs()
            samples.append(
                AttackSampleTask(
                    client_id=client.client_id,
                    round_index=round_index,
                    sample_index=sample_index,
                    target_type=attack_target_type,
                    round_base_state=_clone_state(round_base_state),
                    target=_clone_attack_target(target),
                    real_x=real_x.detach().cpu().clone(),
                    real_y=real_y.detach().cpu().clone(),
                    reference_inputs=None if reference_inputs is None else reference_inputs.detach().cpu().clone(),
                )
            )
    return AttackRoundTask(
        round_index=round_index,
        clients_this_round=len(selected_clients),
        samples_per_client=sample_count,
        samples=samples,
    )


def _execute_attack_round_task(
    config: dict[str, Any],
    task: AttackRoundTask,
    attack_device: torch.device,
) -> AttackRoundResult:
    """Run one detached round of DLG/iDLG evaluation from frozen snapshots."""

    start = time.perf_counter()
    attacks = []
    for sample in task.samples:
        attacks.extend([
            dlg_attack(
                config,
                sample.round_base_state,
                sample.target,
                sample.real_x,
                sample.real_y,
                attack_device,
                target_type=sample.target_type,
                reference_inputs=sample.reference_inputs,
            ),
            idlg_attack(
                config,
                sample.round_base_state,
                sample.target,
                sample.real_x,
                sample.real_y,
                attack_device,
                target_type=sample.target_type,
                reference_inputs=sample.reference_inputs,
            ),
        ])
    return AttackRoundResult(
        round_index=task.round_index,
        time_seconds=time.perf_counter() - start,
        clients_this_round=task.clients_this_round,
        samples_per_client=task.samples_per_client,
        attacks=attacks,
    )


def _round_attack_payload(round_result: AttackRoundResult, cumulative_results: list[Any]) -> dict[str, float]:
    """Build per-round and cumulative attack metrics for tracking/logging."""

    round_attacks = round_result.attacks
    payload: dict[str, float] = {
        "attack/time_seconds": round_result.time_seconds,
        "attack/evaluations_this_round": float(len(round_attacks)),
        "attack/clients_this_round": float(round_result.clients_this_round),
        "attack/samples_per_client": float(round_result.samples_per_client),
        "attack/overall_avg_mse_so_far": sum(result.mse for result in cumulative_results) / len(cumulative_results),
        "attack/success_rate_so_far": attack_success_rate(cumulative_results),
    }
    for name in sorted({result.name for result in round_attacks}):
        subset = [result for result in round_attacks if result.name == name]
        prefix = f"attack/{name}"
        payload[f"{prefix}/mse"] = sum(result.mse for result in subset) / len(subset)
        payload[f"{prefix}/reconstruction_mse"] = payload[f"{prefix}/mse"]
        cumulative_subset = [result for result in cumulative_results if result.name == name]
        payload[f"{prefix}/avg_mse_so_far"] = sum(result.mse for result in cumulative_subset) / len(cumulative_subset)
        payload[f"{prefix}/psnr"] = sum(result.psnr for result in subset) / len(subset)
        payload[f"{prefix}/ssim"] = sum(result.ssim for result in subset) / len(subset)
        payload[f"{prefix}/iterations"] = float(subset[0].iterations)
        payload[f"{prefix}/time_seconds"] = sum(result.time_seconds for result in subset) / len(subset)
        payload[f"{prefix}/gradient_mse"] = sum(result.gradient_mse for result in subset) / len(subset)
        payload[f"{prefix}/success"] = sum(float(result.success) for result in subset) / len(subset)
        payload[f"{prefix}/success_rate_so_far"] = attack_success_rate(cumulative_results, name)
    return payload


class AsyncAttackManager:
    """Queue attack evaluations away from the training hot path and drain them in order."""

    def __init__(self, config: dict[str, Any], tracker: Tracker):
        self.config = config
        self.tracker = tracker
        self.attack_results: list[Any] = []
        self.attack_device = _resolve_attack_device(config)
        attack_cfg = config.get("attack", {})
        self.async_enabled = bool(attack_cfg.get("async_enabled", False))
        self.pending_round_order: list[int] = []
        self.pending_futures: dict[int, Future] = {}
        self.completed_rounds: dict[int, AttackRoundResult] = {}
        self.executor: ThreadPoolExecutor | None = None
        self.async_workers = max(1, int(attack_cfg.get("async_workers", 1)))
        configured_pending = attack_cfg.get("async_max_pending_rounds", 5)
        self.max_pending_rounds = max(self.async_workers, int(configured_pending))
        if attack_cfg.get("enabled", True) and self.async_enabled:
            self.executor = ThreadPoolExecutor(max_workers=self.async_workers, thread_name_prefix="attack")
            logger.info(
                "Async attack manager enabled with workers={} max_pending_rounds={} device={}",
                self.async_workers,
                self.max_pending_rounds,
                self.attack_device,
            )

    def _inflight_rounds(self) -> int:
        """Return the number of attack rounds that still occupy queue capacity."""

        return len(self.pending_round_order)

    def _wait_for_capacity(self) -> None:
        """Block only when the bounded async queue is full."""

        if self.executor is None:
            return
        while self._inflight_rounds() >= self.max_pending_rounds:
            logger.info(
                "Async attack queue full inflight_rounds={}/{}; waiting for one round to finish",
                self._inflight_rounds(),
                self.max_pending_rounds,
            )
            if self.pending_futures:
                done, _ = wait(tuple(self.pending_futures.values()), return_when=FIRST_COMPLETED)
                if not done:
                    break
            self.drain_completed(wait=False)

    def submit(self, task: AttackRoundTask | None) -> None:
        """Run or enqueue one round of attack work."""

        if task is None:
            return
        if self.executor is not None:
            self._wait_for_capacity()
        self.pending_round_order.append(task.round_index)
        if self.executor is None:
            self.completed_rounds[task.round_index] = _execute_attack_round_task(self.config, task, self.attack_device)
        else:
            self.pending_futures[task.round_index] = self.executor.submit(
                _execute_attack_round_task,
                self.config,
                task,
                self.attack_device,
            )
            logger.info(
                "Round {} submitted async attack job with {} evaluations on {} queue_depth={}/{}",
                task.round_index,
                len(task.samples) * 2,
                self.attack_device,
                self._inflight_rounds(),
                self.max_pending_rounds,
            )
        self.drain_completed(wait=False)

    def drain_completed(self, wait: bool = False) -> None:
        """Collect finished futures and log any now-complete rounds in round order."""

        if wait:
            for round_index, future in list(self.pending_futures.items()):
                self.completed_rounds[round_index] = future.result()
                del self.pending_futures[round_index]
        else:
            for round_index, future in list(self.pending_futures.items()):
                if future.done():
                    self.completed_rounds[round_index] = future.result()
                    del self.pending_futures[round_index]
        while self.pending_round_order and self.pending_round_order[0] in self.completed_rounds:
            round_index = self.pending_round_order.pop(0)
            round_result = self.completed_rounds.pop(round_index)
            self.attack_results.extend(round_result.attacks)
            payload = _round_attack_payload(round_result, self.attack_results)
            self.tracker.log(payload, step=round_index)
            logger.info("Round {} attack metrics {}", round_index, payload)

    def finalize(self) -> None:
        """Wait for outstanding attack work and shut down the executor."""

        if self.pending_futures:
            logger.info("Waiting for {} async attack rounds to finish", len(self.pending_futures))
        self.drain_completed(wait=True)
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None


def run_centralized(config: dict[str, Any]) -> dict[str, float]:
    """Run centralized training over all client datasets."""

    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir, config.get("runtime", {}).get("log_level", "INFO"))
    configure_torch_runtime(config)
    configure_random_seed(config)
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
    """Run single-process federated training."""

    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir, config.get("runtime", {}).get("log_level", "INFO"))
    configure_torch_runtime(config)
    configure_random_seed(config)
    tracker = Tracker(config)
    start_time = time.perf_counter()
    device = resolve_device(config)
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    server = FederatedServer(config, val_loader, test_loader, device)
    clients = [FederatedClient(client_id, loader, config, device) for client_id, loader in train_loaders.items()]
    compressed = is_compressed_algorithm(config)
    algorithm = str(config["federated"].get("algorithm", "fedavg"))
    attack_target_type = _attack_target_type(config)
    logger.info(
        "Starting federated run algorithm={} clients={} compressed_uploads={} attack_target_type={}",
        algorithm,
        [client.client_id for client in clients],
        compressed,
        attack_target_type,
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
    attack_manager = AsyncAttackManager(config, tracker)
    for round_index in range(max_rounds):
        round_start = time.perf_counter()
        round_base_state = _clone_state(server.global_state)
        results = [client.train(round_base_state, compressed=compressed, round_index=round_index) for client in clients]
        if compressed:
            aggregation_weights = server.aggregate_sparse(results)
        else:
            aggregation_weights = server.aggregate_dense(results)
        metrics = server.evaluate_global()
        record = server.record_round(
            round_index,
            results,
            aggregation_weights,
            metrics,
            round_time_seconds=time.perf_counter() - round_start,
            elapsed_time_seconds=time.perf_counter() - start_time,
        )
        tracker.log({**_wandb_round_payload(record), **_wandb_cumulative_communication_payload(server.history)}, step=round_index)
        attack_task = _build_attack_round_task(
            config,
            clients,
            results,
            round_index,
            max_rounds,
            round_base_state,
            attack_target_type,
        )
        attack_manager.submit(attack_task)
        if stopper.update(metrics["mse"]):
            logger.info("Early stopping at round {}", round_index)
            break
    test_metrics = server.test_global()
    attack_manager.finalize()
    total_elapsed = time.perf_counter() - start_time
    server.save(output_dir, config)
    tracker.log({**{f"test/{key}": value for key, value in test_metrics.items()}, "run/total_time_seconds": total_elapsed})
    tracker.finish()
    attack_records = [result.to_record() for result in attack_manager.attack_results]
    with (output_dir / "attack_results.json").open("w", encoding="utf-8") as handle:
        json.dump(attack_records, handle, ensure_ascii=False, indent=2)
    attack_summary = summarize_attack_results(
        attack_manager.attack_results,
        float(config.get("attack", {}).get("success_rate_threshold", 0.03)),
    )
    summary = {
        "test": test_metrics,
        "rounds": len(server.history),
        "total_time_seconds": total_elapsed,
        "last_upload_compression_ratio": server.history[-1].upload_compression_ratio if server.history else 0.0,
        "last_total_communication_ratio": server.history[-1].total_communication_ratio if server.history else 0.0,
        "last_communication_ratio": server.history[-1].communication_ratio if server.history else 0.0,
        **_round_history_communication_summary(server.history),
        "attack_target_type": attack_summary.get("target_type", attack_target_type),
        "attack_primary_metric": attack_summary["primary_metric"],
        "attack_primary_metric_direction": attack_summary["primary_metric_direction"],
        "attack_overall_avg_mse": attack_summary["overall_avg_mse"],
        "attack_success_rate": attack_summary["overall_success_rate"],
        "attack_evaluations": len(attack_records),
        "attack_summary": attack_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("Finished experiment: {}", summary)
    return summary
