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

from fedlab.utils.artifacts import save_experiment_config, save_federated_snapshot, should_save_periodic_artifacts
from fedlab.security.attacks import (
    attack_success_rate,
    attach_attack_metadata,
    dlg_attack,
    idlg_attack,
    save_attack_artifacts,
    summarize_attack_results,
)
from fedlab.federated.client import FederatedClient
from fedlab.federated.methods import build_method, is_registered_compressed
from fedlab.datasets import build_federated_loaders
from fedlab.utils.logging import setup_logging
from fedlab.modeling import build_model
from fedlab.utils.serialization import (
    StateDict,
    decompress_topk,
    dequantize_state_update,
    serialize_model,
    state_num_bytes,
    state_num_parameters,
)
from fedlab.federated.server import EarlyStopper, FederatedServer, RoundRecord
from fedlab.federated.protocol import validate_transport_modes
from fedlab.utils.tracking import Tracker
from fedlab.engine.training import build_training_optimizer, evaluate, predict_first_batch, predict_first_batch_for_state, train_one_epoch


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


def _resolve_centralized_rounds(config: dict[str, Any]) -> int:
    """Resolve centralized training rounds from the dedicated config block."""

    centralized_cfg = config.get("centralized", {})
    if centralized_cfg.get("rounds") is None:
        return 10
    return int(centralized_cfg.get("rounds"))


def _log_mode_specific_schedule(config: dict[str, Any], mode: str) -> None:
    """Log when a mode is carrying schedule keys that do not apply to it."""

    if mode == "centralized":
        rounds = config.get("federated", {}).get("rounds")
        if rounds is not None:
            logger.info("Centralized mode ignores federated.rounds={} and uses centralized.rounds", rounds)
    elif mode == "federated":
        centralized_cfg = config.get("centralized", {})
        central_rounds = centralized_cfg.get("rounds")
        if central_rounds is not None:
            logger.info("Federated mode ignores centralized.rounds={} and uses federated.rounds", central_rounds)


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve the configured torch device with a CPU fallback."""

    requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    return torch.device(requested)


def _loader_num_samples(loader: Any) -> int:
    """Return the number of samples carried by one loader-like object."""

    dataset = getattr(loader, "dataset", None)
    return len(dataset) if dataset is not None else len(loader)


def is_compressed_algorithm(config: dict[str, Any]) -> bool:
    """Return whether the configured FL algorithm compresses client uploads."""

    algorithm = str(config.get("federated", {}).get("algorithm", "fedavg")).lower()
    return is_registered_compressed(algorithm)


def _should_capture_update_payload(config: dict[str, Any], round_index: int, max_rounds: int) -> bool:
    """Return whether this round should persist server-visible client updates."""

    attack_cfg = config.get("attack", {})
    frequency = int(attack_cfg.get("frequency_rounds", 1))
    return round_index == 0 or round_index == max_rounds - 1 or (frequency > 0 and round_index % frequency == 0)


def _should_run_attack(config: dict[str, Any], round_index: int, max_rounds: int) -> bool:
    """Return whether attack evaluation should run on this round."""

    attack_cfg = config.get("attack", {})
    if not attack_cfg.get("enabled", True):
        return False
    return _should_capture_update_payload(config, round_index, max_rounds)


def _select_attack_clients(clients: list[FederatedClient], config: dict[str, Any], round_index: int) -> list[FederatedClient]:
    """Select which clients should be attacked on the current round."""

    attack_cfg = config.get("attack", {})
    selection = str(attack_cfg.get("client_selection", "all")).lower()
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
    if "round/val_mse" in payload:
        payload["round/active_val_mse"] = payload["round/val_mse"]
        payload["round/active_val_mae"] = payload.get("round/val_mae")
        payload["round/active_val_mape"] = payload.get("round/val_mape")
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
            "last_parameter_download_compression_ratio": 0.0,
            "last_parameter_upload_compression_ratio": 0.0,
            "last_parameter_total_communication_ratio": 0.0,
            "last_transport_download_compression_ratio": 0.0,
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
        "last_parameter_download_compression_ratio": last.parameter_download_compression_ratio,
        "last_parameter_upload_compression_ratio": last.parameter_upload_compression_ratio,
        "last_parameter_total_communication_ratio": last.parameter_total_communication_ratio,
        "last_transport_download_compression_ratio": last.transport_download_compression_ratio,
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


def _build_federated_summary(
    *,
    server: FederatedServer,
    test_metrics: dict[str, float],
    total_elapsed: float,
    best_round: int,
    best_metrics: dict[str, float],
    attack_records: list[dict[str, Any]],
    attack_summary: dict[str, Any],
    attack_target_type: str,
    protocol_test_metrics: dict[str, float] | None = None,
    oracle_test_metrics: dict[str, float] | None = None,
    transport: str | None = None,
) -> dict[str, Any]:
    """Build a summary payload shared by final outputs and periodic snapshots."""

    history = server.history
    last_privacy = history[-1] if history else None
    protocol_test = test_metrics if protocol_test_metrics is None else protocol_test_metrics
    oracle_test = protocol_test if oracle_test_metrics is None else oracle_test_metrics
    summary = {
        "test": test_metrics,
        "active_test_scope": server.evaluation_mode,
        "evaluation_mode": server.evaluation_mode,
        "best_val_scope": server.evaluation_mode,
        "protocol_test": protocol_test,
        "oracle_test": oracle_test,
        "rounds": len(history),
        "total_time_seconds": total_elapsed,
        "best_round": best_round,
        "best_val_mse": best_metrics["mse"],
        "best_val_mae": best_metrics["mae"],
        "best_val_mape": best_metrics["mape"],
        "test_checkpoint": "best_validation",
        "last_parameter_download_compression_ratio": history[-1].parameter_download_compression_ratio if history else 0.0,
        "last_parameter_upload_compression_ratio": history[-1].parameter_upload_compression_ratio if history else 0.0,
        "last_parameter_total_communication_ratio": history[-1].parameter_total_communication_ratio if history else 0.0,
        **_round_history_communication_summary(history),
        "attack_target_type": attack_summary.get("target_type", attack_target_type),
        "attack_primary_metric_name": attack_summary["primary_metric_name"],
        "attack_primary_metric_direction": attack_summary["primary_metric_direction"],
        "attack_overall_avg_primary_metric_value": attack_summary["overall_avg_primary_metric_value"],
        "attack_overall_best_primary_metric_value": attack_summary["overall_best_primary_metric_value"],
        "attack_success_rate": attack_summary["overall_success_rate"],
        "attack_evaluations": len(attack_records),
        "attack_summary": attack_summary,
        "privacy_accountant": None if last_privacy is None else last_privacy.privacy_accountant,
        "privacy_epsilon": None if last_privacy is None else last_privacy.privacy_epsilon,
        "privacy_delta": None if last_privacy is None else last_privacy.privacy_delta,
        "privacy_rdp_alpha": None if last_privacy is None else last_privacy.privacy_rdp_alpha,
        "privacy_rdp_total": None if last_privacy is None else last_privacy.privacy_rdp_total,
        "privacy_sampling_rate": None if last_privacy is None else last_privacy.privacy_sampling_rate,
        "adaptive_clip_norm": None if last_privacy is None else last_privacy.adaptive_clip_norm,
        "adaptive_clip_median_norm": None if last_privacy is None else last_privacy.adaptive_clip_median_norm,
        "adaptive_reference_clip_norm": None if last_privacy is None else last_privacy.adaptive_reference_clip_norm,
        "adaptive_noise_std": None if last_privacy is None else last_privacy.adaptive_noise_std,
        "privacy_trust_model": "central_dp_trusted_aggregator" if (last_privacy is not None and last_privacy.privacy_accountant is not None) else None,
    }
    if transport is not None:
        summary["transport"] = transport
    return summary


def _build_federated_resume_state(
    *,
    round_index: int,
    server: FederatedServer,
    best_global_state: StateDict,
    best_oracle_state: StateDict | None,
    best_metrics: dict[str, float],
    best_round: int,
) -> dict[str, Any]:
    """Capture the minimal federated state needed to continue a stopped run."""

    return {
        "round_index": int(round_index),
        "global_state": _clone_state(server.global_state),
        "oracle_global_state": None if server.oracle_global_state is None else _clone_state(server.oracle_global_state),
        "best_global_state": _clone_state(best_global_state),
        "best_oracle_state": None if best_oracle_state is None else _clone_state(best_oracle_state),
        "best_metrics": dict(best_metrics),
        "best_round": int(best_round),
        "history": [asdict(record) for record in server.history],
    }


def _save_periodic_federated_snapshot(
    *,
    output_dir: Path,
    config: dict[str, Any],
    server: FederatedServer,
    round_index: int,
    start_time: float,
    best_global_state: StateDict,
    best_oracle_state: StateDict | None,
    best_metrics: dict[str, float],
    best_round: int,
    attack_results: list[Any],
    attack_target_type: str,
    transport: str | None = None,
) -> None:
    """Persist a periodic round snapshot without waiting for unfinished async attacks."""

    if not should_save_periodic_artifacts(config, round_index + 1):
        return
    test_metrics = server.test_global()
    protocol_test_metrics, oracle_test_metrics = _resolve_test_metric_views(server, test_metrics)
    attack_records = [result.to_record() for result in attack_results]
    attack_summary = summarize_attack_results(
        attack_results,
        float(config.get("attack", {}).get("success_rate_threshold", 0.03)),
    )
    summary = _build_federated_summary(
        server=server,
        test_metrics=test_metrics,
        total_elapsed=time.perf_counter() - start_time,
        best_round=best_round,
        best_metrics=best_metrics,
        attack_records=attack_records,
        attack_summary=attack_summary,
        attack_target_type=attack_target_type,
        protocol_test_metrics=protocol_test_metrics,
        oracle_test_metrics=oracle_test_metrics,
        transport=transport,
    )
    snapshot_dir = save_federated_snapshot(
        output_dir,
        config,
        snapshot_name=f"round_{round_index + 1:04d}",
        model_state=server.global_state,
        oracle_model_state=server.oracle_global_state if server._uses_oracle_evaluation() else None,
        metrics_history=server.history,
        summary=summary,
        attack_records=attack_records,
        resume_state=_build_federated_resume_state(
            round_index=round_index + 1,
            server=server,
            best_global_state=best_global_state,
            best_oracle_state=best_oracle_state,
            best_metrics=best_metrics,
            best_round=best_round,
        ),
    )
    logger.info("Saved periodic snapshot at round {} to {}", round_index, snapshot_dir)


def _wandb_cumulative_communication_payload(history: list[RoundRecord]) -> dict[str, float | int]:
    """Return cumulative communication metrics in a wandb-friendly flat namespace."""

    summary = _round_history_communication_summary(history)
    return {f"cumulative/{key}": value for key, value in summary.items()}


def _resolve_test_metric_views(server, active_test_metrics):
    """Return protocol/oracle test metrics, defaulting oracle to protocol when not configured."""

    protocol_test_metrics = server.test_protocol() if server._uses_oracle_evaluation() else active_test_metrics
    oracle_test_metrics = server.test_oracle() if server._uses_oracle_evaluation() else protocol_test_metrics
    return protocol_test_metrics, oracle_test_metrics


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
    if not protected or algorithm in {"fedavg", "fedaware"}:
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
    if algorithm == "sign_fedavg":
        scale = max(float(flat.abs().mean().item()), 1e-12)
        flat = torch.sign(flat).to(torch.float32) * scale
    if algorithm == "qsgd_fedavg":
        levels = int(config.get("federated", {}).get("qsgd_levels", 127))
        norm = max(float(torch.linalg.vector_norm(flat).item()), 1e-12)
        scaled = torch.clamp(flat.abs() / norm, 0.0, 1.0) * float(levels)
        lower = torch.floor(scaled)
        probability = scaled - lower
        random = torch.rand(scaled.shape, generator=generator, dtype=torch.float32) if generator is not None else torch.rand(scaled.shape, dtype=torch.float32)
        bucket = torch.clamp(lower + (random < probability).to(torch.float32), 0.0, float(levels))
        flat = torch.sign(flat).to(torch.float32) * bucket * (norm / float(levels))

    if algorithm in {"compressed_fedavg", "sparse_fedavg", "dp_topk_fedavg", "soteriafl", "randomk_fedavg"}:
        fraction = float(config.get("federated", {}).get("topk_fraction", 0.05))
        total = flat.numel()
        k = max(1, int(total * fraction))
        sparse = torch.zeros_like(flat)
        if algorithm in {"soteriafl", "randomk_fedavg"}:
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


def _update_best_checkpoint(
    best_state: StateDict | None,
    best_metrics: dict[str, float] | None,
    best_index: int | None,
    candidate_state: StateDict,
    candidate_metrics: dict[str, float],
    candidate_index: int,
    label: str,
    metric_name: str = "mse",
    metric_mode: str = "min",
) -> tuple[StateDict, dict[str, float], int, bool]:
    """Track the best validation checkpoint seen so far.

    Example:
        ``best_state, best_metrics, best_round, improved = _update_best_checkpoint(...)``
        stores a detached clone of the current model when validation MSE improves.
    """

    candidate_metrics = {key: float(value) for key, value in candidate_metrics.items()}
    improved = best_metrics is None or (candidate_metrics[metric_name] < float(best_metrics[metric_name]) if metric_mode == "min" else candidate_metrics[metric_name] > float(best_metrics[metric_name]))
    if improved:
        logger.info(
            "New best validation checkpoint for {} at {}={} {}={:.6f}",
            label,
            label,
            candidate_index,
            metric_name,
            candidate_metrics[metric_name],
        )
        return _clone_state(candidate_state), candidate_metrics, candidate_index, True
    return _clone_state(best_state), dict(best_metrics), int(best_index), False


def _attack_target_type(config: dict[str, Any]) -> str:
    """Return the configured interception target for reconstruction attacks."""

    return str(config.get("attack", {}).get("target_type", "update_payload")).lower()


def _extract_attack_payload(
    config: dict[str, Any],
    result,
    results,
    server: FederatedServer | None = None,
    round_base_state: StateDict | None = None,
    round_index: int = 0,
    round_context: dict[str, Any] | None = None,
) -> StateDict:
    """Return the actual transmitted client payload as a dense state update."""

    algorithm = str(config.get("federated", {}).get("algorithm", "fedavg")).lower()
    method = build_method(algorithm)
    return method.extract_attack_payload(
        result=result,
        results=results,
        server=server,
        clone_state=_clone_state,
        round_base_state=round_base_state,
        round_index=round_index,
        round_context=round_context or {},
    )


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
    reference_targets: torch.Tensor | None = None
    scale_mean: list[float] | None = None
    scale_std: list[float] | None = None


@dataclass
class AttackRoundTask:
    """One round of attack work detached from the training hot path."""

    round_index: int
    clients_this_round: int
    evaluations_per_client: int
    samples: list[AttackSampleTask]


@dataclass
class AttackRoundResult:
    """Completed attack artifacts for one training round."""

    round_index: int
    time_seconds: float
    clients_this_round: int
    evaluations_per_client: int
    attacks: list[Any]


def _clone_attack_target(target: list[torch.Tensor] | StateDict) -> list[torch.Tensor] | StateDict:
    """Detach and clone an intercepted attack target onto CPU memory."""

    if isinstance(target, list):
        return [tensor.detach().cpu().clone() for tensor in target]
    return _clone_state(target)


def _select_attack_client_ids(client_ids: list[str], config: dict[str, Any], round_index: int) -> list[str]:
    """Return the configured attack client subset while preserving client order."""

    if not client_ids:
        return []
    attack_cfg = config.get("attack", {})
    selection = str(attack_cfg.get("client_selection", "all")).lower()
    count = max(1, int(attack_cfg.get("clients_per_round", 1)))
    if selection == "all":
        return list(client_ids)
    if selection == "first":
        return list(client_ids[:count])
    start = round_index % len(client_ids)
    return [client_ids[(start + offset) % len(client_ids)] for offset in range(min(count, len(client_ids)))]


def _capture_round_update_records(
    config: dict[str, Any],
    clients: list[FederatedClient],
    results,
    round_index: int,
    max_rounds: int,
    round_base_state: StateDict,
    server: FederatedServer | None = None,
    round_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Capture server-visible client updates plus replay inputs for one round."""

    if not _should_capture_update_payload(config, round_index, max_rounds):
        return []
    attack_cfg = config.get("attack", {})
    max_samples = int(attack_cfg.get("max_samples", 1))
    sample_count = max(1, int(attack_cfg.get("sample_count", 1)))
    round_context = round_context or {}
    results_by_client = {result.client_id: result for result in results}
    records: list[dict[str, Any]] = []
    for client in clients:
        result = results_by_client[client.client_id]
        target_update = _extract_attack_payload(
            config,
            result,
            results,
            server=server,
            round_base_state=round_base_state,
            round_index=round_index,
            round_context=round_context,
        )
        train_loader = getattr(client, "train_loader", None)
        scaler = getattr(train_loader, "scaler", None)
        reference_inputs = client.train_reference_inputs()
        reference_target_getter = getattr(client, "train_reference_targets", None)
        reference_targets = reference_target_getter() if callable(reference_target_getter) else None
        samples = []
        for sample_index in range(sample_count):
            real_x, real_y = client.sample_batch(max_samples=max_samples, batch_index=sample_index)
            samples.append(
                {
                    "sample_index": int(sample_index),
                    "real_x": real_x.detach().cpu().clone(),
                    "real_y": real_y.detach().cpu().clone(),
                }
            )
        records.append(
            {
                "client_id": client.client_id,
                "round_index": int(round_index),
                "target_type": "update_payload",
                "aggregation_payload_kind": getattr(result, "aggregation_payload_kind", "dense_update"),
                "compressor": getattr(result, "compressor", "none"),
                "upload_mode": getattr(result, "upload_mode", "update"),
                "round_base_state": _clone_state(round_base_state),
                "target_update": _clone_state(target_update),
                "reference_inputs": None if reference_inputs is None else reference_inputs.detach().cpu().clone(),
                "reference_targets": None if reference_targets is None else reference_targets.detach().cpu().clone(),
                "scale_mean": None if getattr(scaler, "mean", None) is None else [float(value) for value in scaler.mean.reshape(-1).tolist()],
                "scale_std": None if getattr(scaler, "std", None) is None else [float(value) for value in scaler.std.reshape(-1).tolist()],
                "samples": samples,
            }
        )
    return records


def save_captured_update_records(output_dir: Path, records: list[dict[str, Any]]) -> list[Path]:
    """Persist captured server-visible updates under per-client subdirectories."""

    if not records:
        return []
    capture_root = output_dir / "saved_updates"
    capture_root.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for record in sorted(records, key=lambda item: (str(item["client_id"]), int(item["round_index"]))):
        client_id = str(record["client_id"])
        round_index = int(record["round_index"])
        relative_path = Path(client_id) / f"round_{round_index:04d}.pt"
        path = capture_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, path)
        saved_paths.append(path)
    index: list[dict[str, Any]] = []
    for path in sorted(capture_root.rglob("round_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        index.append(
            {
                "client_id": str(payload["client_id"]),
                "round_index": int(payload["round_index"]),
                "target_type": payload.get("target_type", "update_payload"),
                "path": str(path.relative_to(output_dir)),
            }
        )
    with (capture_root / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
    return saved_paths


def load_captured_update_records(run_dir: Path) -> list[dict[str, Any]]:
    """Load persisted per-client update captures sorted by round and client."""

    capture_root = Path(run_dir) / "saved_updates"
    if not capture_root.exists():
        return []
    index_path = capture_root / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        records = [torch.load(Path(run_dir) / entry["path"], map_location="cpu", weights_only=False) for entry in index]
    else:
        records = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(capture_root.rglob("round_*.pt"))]
    return sorted(records, key=lambda item: (int(item["round_index"]), str(item["client_id"])))


def build_update_attack_round_task(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    round_index: int,
    max_rounds: int,
) -> AttackRoundTask | None:
    """Build one update-payload attack task from previously captured update records."""

    if not _should_run_attack(config, round_index, max_rounds):
        return None
    records_this_round = [record for record in records if int(record["round_index"]) == int(round_index)]
    if not records_this_round:
        return None
    client_order = [str(client_id) for client_id in config.get("data", {}).get("clients", [])]
    order_index = {client_id: index for index, client_id in enumerate(client_order)}
    records_this_round = sorted(
        records_this_round,
        key=lambda record: (order_index.get(str(record["client_id"]), len(order_index)), str(record["client_id"])),
    )
    client_ids = [str(record["client_id"]) for record in records_this_round]
    selected_client_ids = set(_select_attack_client_ids(client_ids, config, round_index))
    selected_records = [record for record in records_this_round if str(record["client_id"]) in selected_client_ids]
    if not selected_records:
        return None
    samples: list[AttackSampleTask] = []
    evaluations_per_client = 0
    for record in selected_records:
        record_samples = list(record.get("samples", []))
        evaluations_per_client = max(evaluations_per_client, len(record_samples))
        for sample in record_samples:
            samples.append(
                AttackSampleTask(
                    client_id=str(record["client_id"]),
                    round_index=int(record["round_index"]),
                    sample_index=int(sample.get("sample_index", 0)),
                    target_type="update_payload",
                    round_base_state=_clone_state(record["round_base_state"]),
                    target=_clone_state(record["target_update"]),
                    real_x=sample["real_x"].detach().cpu().clone(),
                    real_y=sample["real_y"].detach().cpu().clone(),
                    reference_inputs=None if record.get("reference_inputs") is None else record["reference_inputs"].detach().cpu().clone(),
                    reference_targets=None if record.get("reference_targets") is None else record["reference_targets"].detach().cpu().clone(),
                    scale_mean=None if record.get("scale_mean") is None else [float(value) for value in record["scale_mean"]],
                    scale_std=None if record.get("scale_std") is None else [float(value) for value in record["scale_std"]],
                )
            )
    return AttackRoundTask(
        round_index=int(round_index),
        clients_this_round=len(selected_records),
        evaluations_per_client=evaluations_per_client,
        samples=samples,
    )


def _resolve_attack_device(config: dict[str, Any]) -> torch.device:
    """Resolve the device used for reconstruction attacks."""

    attack_cfg = config.get("attack", {})
    requested = str(attack_cfg.get("device", "same")).lower()
    if requested == "same":
        requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Attack device {} unavailable; falling back to CPU", requested)
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
    server: FederatedServer | None = None,
    round_context: dict[str, Any] | None = None,
) -> AttackRoundTask | None:
    """Capture one round of attack inputs as immutable CPU snapshots."""

    if not _should_run_attack(config, round_index, max_rounds):
        return None
    attack_cfg = config.get("attack", {})
    max_samples = int(attack_cfg.get("max_samples", 1))
    sample_count = max(1, int(attack_cfg.get("sample_count", 1)))
    selected_clients = _select_attack_clients(clients, config, round_index)
    round_context = round_context or {}
    results_by_client = {result.client_id: result for result in results}
    samples: list[AttackSampleTask] = []
    for client_index, client in enumerate(selected_clients):
        result = results_by_client[client.client_id]
        for sample_index in range(sample_count):
            reference_inputs = None
            reference_targets = None
            train_loader = getattr(client, "train_loader", None)
            scaler = getattr(train_loader, "scaler", None)
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
                target = _extract_attack_payload(
                    config,
                    result,
                    results,
                    server=server,
                    round_base_state=round_base_state,
                    round_index=round_index,
                    round_context=round_context,
                )
                reference_inputs = client.train_reference_inputs()
                reference_target_getter = getattr(client, "train_reference_targets", None)
                if callable(reference_target_getter):
                    reference_targets = reference_target_getter()
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
                    reference_targets=None if reference_targets is None else reference_targets.detach().cpu().clone(),
                    scale_mean=None if getattr(scaler, "mean", None) is None else [float(value) for value in scaler.mean.reshape(-1).tolist()],
                    scale_std=None if getattr(scaler, "std", None) is None else [float(value) for value in scaler.std.reshape(-1).tolist()],
                )
            )
    return AttackRoundTask(
        round_index=round_index,
        clients_this_round=len(selected_clients),
        evaluations_per_client=sample_count,
        samples=samples,
    )


def _inverse_plot_tensor(values: torch.Tensor, mean: list[float] | None, std: list[float] | None) -> torch.Tensor:
    """Restore one standardized tensor for visualization only."""

    if mean is None or std is None:
        return values.detach().cpu().clone()
    tensor = values.detach().cpu().to(torch.float32)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).reshape(1, 1, -1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).reshape(1, 1, -1)
    while mean_tensor.ndim < tensor.ndim:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    return tensor * std_tensor + mean_tensor


def _execute_attack_round_task(
    config: dict[str, Any],
    task: AttackRoundTask,
    attack_device: torch.device,
) -> AttackRoundResult:
    """Run one detached round of DLG/iDLG evaluation from frozen snapshots."""

    start = time.perf_counter()
    attacks = []
    for sample in task.samples:
        for result in (
            attach_attack_metadata(
                dlg_attack(
                    config,
                    sample.round_base_state,
                    sample.target,
                    sample.real_x,
                    sample.real_y,
                    attack_device,
                    target_type=sample.target_type,
                    reference_inputs=sample.reference_inputs,
                    reference_targets=sample.reference_targets,
                ),
                client_id=sample.client_id,
                round_index=sample.round_index,
                sample_index=sample.sample_index,
            ),
            attach_attack_metadata(
                idlg_attack(
                    config,
                    sample.round_base_state,
                    sample.target,
                    sample.real_x,
                    sample.real_y,
                    attack_device,
                    target_type=sample.target_type,
                    reference_inputs=sample.reference_inputs,
                    reference_targets=sample.reference_targets,
                ),
                client_id=sample.client_id,
                round_index=sample.round_index,
                sample_index=sample.sample_index,
            ),
        ):
            plot_reference_x = result.reference_x if getattr(result, "reference_x", None) is not None else result.real_x
            plot_reference_y = result.reference_y if getattr(result, "reference_y", None) is not None else result.real_y
            result.plot_reference_x = _inverse_plot_tensor(plot_reference_x, sample.scale_mean, sample.scale_std)
            result.plot_reconstructed_x = _inverse_plot_tensor(result.reconstructed_x, sample.scale_mean, sample.scale_std)
            result.plot_reference_y = None if plot_reference_y is None else _inverse_plot_tensor(plot_reference_y, sample.scale_mean, sample.scale_std)
            result.plot_reconstructed_y = None if result.reconstructed_y is None else _inverse_plot_tensor(result.reconstructed_y, sample.scale_mean, sample.scale_std)
            result.plot_real_x = result.plot_reference_x
            result.plot_real_y = result.plot_reference_y
            attacks.append(result)
    return AttackRoundResult(
        round_index=task.round_index,
        time_seconds=time.perf_counter() - start,
        clients_this_round=task.clients_this_round,
        evaluations_per_client=task.evaluations_per_client,
        attacks=attacks,
    )


def _mean_finite(values: list[float]) -> float | None:
    """Return the mean of finite values, or None when all values are non-finite."""

    finite = [value for value in values if value is not None and isinstance(value, (int, float)) and float("-inf") < float(value) < float("inf")]
    if not values:
        return 0.0
    if not finite:
        return None
    return sum(float(value) for value in finite) / len(finite)


def _attack_payload_metrics(subset: list[Any], cumulative_subset: list[Any], prefix: str) -> dict[str, float | str]:
    """Return one attack-metric payload block for one subset prefix."""

    if not subset:
        return {}
    payload: dict[str, float | str | None] = {}
    payload[f"{prefix}/primary_metric_name"] = getattr(subset[0], "metric_name", "reconstruction_mse")
    payload[f"{prefix}/primary_metric_value"] = sum(result.mse for result in subset) / len(subset)
    payload[f"{prefix}/cumulative_avg_primary_metric_value"] = 0.0 if not cumulative_subset else sum(result.mse for result in cumulative_subset) / len(cumulative_subset)
    exact_values = [getattr(result, "exact_target_mse", None) for result in subset if getattr(result, "exact_target_mse", None) is not None]
    exact_so_far = [getattr(result, "exact_target_mse", None) for result in cumulative_subset if getattr(result, "exact_target_mse", None) is not None]
    nearest_values = [getattr(result, "nearest_client_train_mse", None) for result in subset if getattr(result, "nearest_client_train_mse", None) is not None]
    nearest_so_far = [getattr(result, "nearest_client_train_mse", None) for result in cumulative_subset if getattr(result, "nearest_client_train_mse", None) is not None]
    if exact_values:
        payload[f"{prefix}/exact_target_mse"] = _mean_finite(exact_values)
        payload[f"{prefix}/cumulative_avg_exact_target_mse"] = _mean_finite(exact_so_far)
    if nearest_values:
        payload[f"{prefix}/nearest_client_train_mse"] = _mean_finite(nearest_values)
        payload[f"{prefix}/cumulative_avg_nearest_client_train_mse"] = _mean_finite(nearest_so_far)
    payload[f"{prefix}/psnr"] = sum(result.psnr for result in subset) / len(subset)
    payload[f"{prefix}/ssim"] = sum(result.ssim for result in subset) / len(subset)
    payload[f"{prefix}/iterations"] = float(subset[0].iterations)
    payload[f"{prefix}/time_seconds"] = sum(result.time_seconds for result in subset) / len(subset)
    payload[f"{prefix}/objective_mse"] = sum(result.gradient_mse for result in subset) / len(subset)
    payload[f"{prefix}/cumulative_avg_objective_mse"] = _mean_finite([result.gradient_mse for result in cumulative_subset])
    payload[f"{prefix}/success_fraction"] = sum(float(result.success) for result in subset) / len(subset)
    payload[f"{prefix}/cumulative_success_rate"] = attack_success_rate(cumulative_subset)
    return payload


def _round_attack_payload(round_result: AttackRoundResult, cumulative_results: list[Any]) -> dict[str, float | str]:
    """Build per-round and cumulative attack metrics for tracking/logging."""

    round_attacks = round_result.attacks
    primary_metric_name = "reconstruction_mse" if not round_attacks else getattr(round_attacks[0], "metric_name", "reconstruction_mse")
    overall_avg_primary_metric = 0.0 if not cumulative_results else sum(result.mse for result in cumulative_results) / len(cumulative_results)
    payload: dict[str, float | str | None] = {
        "attack/round_index": float(round_result.round_index),
        "attack/time_seconds": round_result.time_seconds,
        "attack/evaluations_this_round": float(len(round_attacks)),
        "attack/clients_this_round": float(round_result.clients_this_round),
        "attack/evaluations_per_client_this_round": float(round_result.evaluations_per_client),
        "attack/primary_metric_name": primary_metric_name,
        "attack/cumulative_avg_primary_metric_value": overall_avg_primary_metric,
        "attack/cumulative_success_rate": attack_success_rate(cumulative_results),
    }
    for name in sorted({result.name for result in round_attacks}):
        subset = [result for result in round_attacks if result.name == name]
        cumulative_subset = [result for result in cumulative_results if result.name == name]
        payload.update(_attack_payload_metrics(subset, cumulative_subset, f"attack/{name}"))
    for client_id in sorted({str(result.client_id) for result in round_attacks if getattr(result, "client_id", None) is not None}):
        client_subset = [result for result in round_attacks if result.client_id == client_id]
        cumulative_client_subset = [result for result in cumulative_results if result.client_id == client_id]
        client_prefix = f"attack/client/{client_id}"
        payload.update(_attack_payload_metrics(client_subset, cumulative_client_subset, client_prefix))
        for name in sorted({result.name for result in client_subset}):
            method_subset = [result for result in client_subset if result.name == name]
            cumulative_method_subset = [result for result in cumulative_client_subset if result.name == name]
            payload.update(_attack_payload_metrics(method_subset, cumulative_method_subset, f"{client_prefix}/{name}"))
    return payload


def _log_attack_reconstruction_views(tracker: Tracker, results: list[Any], step: int) -> None:
    """Log merged and per-client attack reconstruction figures."""

    if not hasattr(tracker, "log_attack_reconstruction"):
        return
    seen_methods: set[str] = set()
    seen_client_methods: set[tuple[str, str]] = set()
    for result in results:
        method = str(result.name)
        if method not in seen_methods:
            seen_methods.add(method)
            tracker.log_attack_reconstruction(f"attack/{method}/reconstruction", result, step=step)
        client_id = getattr(result, "client_id", None)
        if client_id is None:
            continue
        client_method = (str(client_id), method)
        if client_method in seen_client_methods:
            continue
        seen_client_methods.add(client_method)
        tracker.log_attack_reconstruction(
            f"attack/client/{client_id}/{method}/reconstruction",
            result,
            step=step,
        )


def _iter_client_prediction_loaders(loader: Any, client_ids: list[str] | None) -> list[tuple[str, Any]]:
    """Return ordered client-specific subloaders when the loader is concatenated."""

    if not client_ids:
        return []
    subloaders = getattr(loader, "loaders", None)
    if subloaders is None:
        return []
    if len(subloaders) != len(client_ids):
        logger.debug(
            "Skip split prediction plots because subloader count {} != client count {}",
            len(subloaders),
            len(client_ids),
        )
        return []
    return [(client_id, subloader) for client_id, subloader in zip(client_ids, subloaders)]


def _predict_for_logging(
    model,
    loader: Any,
    device: torch.device,
    state: StateDict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one prediction triplet for merged or client-specific visualization."""

    if state is None:
        return predict_first_batch(model, loader, device)
    return predict_first_batch_for_state(model, state, loader, device)


def _log_prediction_views(
    tracker: Tracker,
    key: str,
    title: str,
    model,
    loader: Any,
    device: torch.device,
    *,
    step: int | None = None,
    client_ids: list[str] | None = None,
    state: StateDict | None = None,
) -> None:
    """Log one merged prediction plot and optional per-client prediction plots."""

    input_series, prediction, target = _predict_for_logging(model, loader, device, state=state)
    tracker.log_prediction_plot(
        key,
        input_series,
        prediction,
        target,
        step=step,
        title=title,
        scaler=getattr(loader, "scaler", None),
    )
    for client_id, client_loader in _iter_client_prediction_loaders(loader, client_ids):
        input_series, prediction, target = _predict_for_logging(model, client_loader, device, state=state)
        tracker.log_prediction_plot(
            f"{key}/client/{client_id}",
            input_series,
            prediction,
            target,
            step=step,
            title=f"{title} client={client_id}",
            scaler=getattr(client_loader, "scaler", getattr(loader, "scaler", None)),
        )


class AsyncAttackManager:
    """Queue attack evaluations away from the training hot path and drain them in order."""

    def __init__(self, config: dict[str, Any], tracker: Tracker):
        """Initialize attack workers and result buffers from config."""

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
            _log_attack_reconstruction_views(self.tracker, round_result.attacks, round_index)
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
    config_formats = config.get("artifacts", {}).get("config_formats")
    saved_configs = save_experiment_config(config, output_dir, config_formats)
    logger.info("Saved startup config artifacts: {}", [str(path) for path in saved_configs])
    configure_torch_runtime(config)
    configure_random_seed(config)
    _log_mode_specific_schedule(config, "centralized")
    tracker = Tracker(config)
    start_time = time.perf_counter()
    device = resolve_device(config)
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    client_ids = list(train_loaders.keys())
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
    optimizer = build_training_optimizer(model.parameters(), config)
    stopper = EarlyStopper(int(config["training"].get("patience", 5)))
    history = []
    best_state = serialize_model(model)
    best_metrics = {"mse": float("inf"), "mae": float("inf"), "mape": float("inf")}
    best_round = -1
    for round_index in range(_resolve_centralized_rounds(config)):
        round_start = time.perf_counter()
        loss = sum(train_one_epoch(model, loader, optimizer, device) for loader in train_loaders.values()) / len(train_loaders)
        metrics = evaluate(model, val_loader, device)
        best_state, best_metrics, best_round, _ = _update_best_checkpoint(
            best_state=best_state,
            best_metrics=best_metrics,
            best_index=best_round,
            candidate_state=serialize_model(model),
            candidate_metrics=metrics,
            candidate_index=round_index,
            label="round",
        )
        round_time = time.perf_counter() - round_start
        elapsed = time.perf_counter() - start_time
        round_record = {
            "round": round_index,
            "train_loss": loss,
            "val_mse": metrics["mse"],
            "val_mae": metrics["mae"],
            "val_mape": metrics["mape"],
            "round_time_seconds": round_time,
            "elapsed_time_seconds": elapsed,
        }
        history.append(round_record)
        logger.info(
            "Centralized round {} loss={:.6f} val_mse={:.6f} time={:.2f}s elapsed={:.2f}s",
            round_index,
            loss,
            metrics["mse"],
            round_time,
            elapsed,
        )
        tracker.log({
            "round/loss": loss,
            "round/val_mse": metrics["mse"],
            "round/val_mae": metrics["mae"],
            "round/val_mape": metrics["mape"],
            "round/time_seconds": round_time,
            "run/elapsed_time_seconds": elapsed,
        }, step=round_index)
        try:
            _log_prediction_views(
                tracker,
                "prediction/centralized/val",
                "centralized val prediction",
                model,
                val_loader,
                device,
                step=round_index,
                client_ids=client_ids,
            )
        except Exception as exc:
            logger.debug("Skip centralized val prediction plot: {}", exc)
        if stopper.update(metrics["mse"]):
            logger.info("Centralized early stopping at round {}", round_index)
            break
    model.load_state_dict(best_state)
    logger.info("Restored best centralized checkpoint from round {} for final test", best_round)
    torch.save(model.state_dict(), output_dir / "model.pt")
    test_metrics = evaluate(model, test_loader, device)
    final_test_step = max(len(history), best_round + 1)
    total_elapsed = time.perf_counter() - start_time
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "test": test_metrics,
                "rounds": len(history),
                "total_time_seconds": total_elapsed,
                "best_round": best_round,
                "best_val": best_metrics,
                "test_checkpoint": "best_validation",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    tracker.log({
        **{f"test/{key}": value for key, value in test_metrics.items()},
        "run/total_time_seconds": total_elapsed,
        "run/best_round": best_round,
        "run/best_val_mse": best_metrics["mse"],
        "run/best_val_mae": best_metrics["mae"],
        "run/best_val_mape": best_metrics["mape"],
    })
    try:
        _log_prediction_views(
            tracker,
            "prediction/centralized/test",
            "centralized test prediction",
            model,
            test_loader,
            device,
            step=final_test_step,
            client_ids=client_ids,
        )
    except Exception as exc:
        logger.debug("Skip centralized prediction plot: {}", exc)
    tracker.finish()
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "test": test_metrics,
                "rounds": len(history),
                "total_time_seconds": total_elapsed,
                "best_round": best_round,
                "best_val_mse": best_metrics["mse"],
                "best_val_mae": best_metrics["mae"],
                "best_val_mape": best_metrics["mape"],
                "test_checkpoint": "best_validation",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Centralized training finished in {:.2f}s with test metrics {}", total_elapsed, test_metrics)
    return test_metrics


def run_federated(config: dict[str, Any]) -> dict[str, Any]:
    """Run single-process federated training."""

    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir, config.get("runtime", {}).get("log_level", "INFO"))
    config_formats = config.get("artifacts", {}).get("config_formats")
    saved_configs = save_experiment_config(config, output_dir, config_formats)
    logger.info("Saved startup config artifacts: {}", [str(path) for path in saved_configs])
    configure_torch_runtime(config)
    configure_random_seed(config)
    validate_transport_modes(config)
    _log_mode_specific_schedule(config, "federated")
    tracker = Tracker(config)
    start_time = time.perf_counter()
    device = resolve_device(config)
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    client_ids = list(train_loaders.keys())
    server = FederatedServer(config, val_loader, test_loader, device)
    total_train_samples = sum(_loader_num_samples(loader) for loader in train_loaders.values())
    clients = [
        FederatedClient(
            client_id,
            loader,
            config,
            device,
            total_train_samples=total_train_samples,
            total_clients=len(train_loaders),
            allow_ega_pretrain=False,
        )
        for client_id, loader in train_loaders.items()
    ]
    compressed = is_compressed_algorithm(config)
    method = build_method(str(config["federated"].get("algorithm", "fedavg")))
    algorithm = method.name
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
    best_global_state = _clone_state(server.global_state)
    best_oracle_state = None if not server._uses_oracle_evaluation() else _clone_state(server.oracle_global_state)
    best_metrics = {"mse": float("inf"), "mae": float("inf"), "mape": float("inf")}
    best_round = -1
    for round_index in range(max_rounds):
        round_start = time.perf_counter()
        round_base_state = _clone_state(server.global_state)
        round_context = server.build_round_context()
        if round_context:
            results = [
                client.train(round_base_state, compressed=compressed, round_index=round_index, round_context=round_context)
                for client in clients
            ]
        else:
            results = [client.train(round_base_state, compressed=compressed, round_index=round_index) for client in clients]
        server.method.sync_server_client_state(server=server, clients=clients)
        if compressed:
            aggregation_weights = server.aggregate_sparse(results, round_base_state=round_base_state, round_index=round_index, round_context=round_context)
        else:
            aggregation_weights = server.aggregate_dense(results, round_index=round_index, round_base_state=round_base_state, round_context=round_context)
        metrics = server.evaluate_global()
        protocol_metrics = server.evaluate_protocol() if server._uses_oracle_evaluation() else metrics
        oracle_metrics = server.evaluate_oracle() if server._uses_oracle_evaluation() else protocol_metrics
        best_global_state, best_metrics, best_round, improved = _update_best_checkpoint(
            best_state=best_global_state,
            best_metrics=best_metrics,
            best_index=best_round,
            candidate_state=server.global_state,
            candidate_metrics=metrics,
            candidate_index=round_index,
            label="round",
        )
        if improved and server._uses_oracle_evaluation():
            best_oracle_state = _clone_state(server.oracle_global_state)
        record = server.record_round(
            round_index,
            results,
            aggregation_weights,
            metrics,
            round_time_seconds=time.perf_counter() - round_start,
            elapsed_time_seconds=time.perf_counter() - start_time,
            protocol_metrics=protocol_metrics,
            oracle_metrics=oracle_metrics,
        )
        tracker.log({**_wandb_round_payload(record), **_wandb_cumulative_communication_payload(server.history)}, step=round_index)
        try:
            _log_prediction_views(
                tracker,
                "prediction/federated/val_protocol",
                "federated val protocol prediction",
                server.model,
                val_loader,
                device,
                step=round_index,
                client_ids=client_ids,
                state=server.global_state,
            )
            oracle_state = server.oracle_global_state if server.oracle_global_state is not None else server.global_state
            _log_prediction_views(
                tracker,
                "prediction/federated/val_oracle",
                "federated val oracle prediction",
                server.model,
                val_loader,
                device,
                step=round_index,
                client_ids=client_ids,
                state=oracle_state,
            )
        except Exception as exc:
            logger.debug("Skip federated val prediction plot: {}", exc)
        captured_update_records = _capture_round_update_records(
            config,
            clients,
            results,
            round_index,
            max_rounds,
            round_base_state,
            server=server,
            round_context=round_context,
        )
        save_captured_update_records(output_dir, captured_update_records)
        if attack_target_type == "update_payload":
            attack_task = build_update_attack_round_task(config, captured_update_records, round_index, max_rounds)
        else:
            attack_task = _build_attack_round_task(
                config,
                clients,
                results,
                round_index,
                max_rounds,
                round_base_state,
                attack_target_type,
                server=server,
                round_context=round_context,
            )
        attack_manager.submit(attack_task)
        _save_periodic_federated_snapshot(
            output_dir=output_dir,
            config=config,
            server=server,
            round_index=round_index,
            start_time=start_time,
            best_global_state=best_global_state,
            best_oracle_state=best_oracle_state,
            best_metrics=best_metrics,
            best_round=best_round,
            attack_results=attack_manager.attack_results,
            attack_target_type=attack_target_type,
        )
        if stopper.update(metrics["mse"]):
            logger.info("Early stopping at round {}", round_index)
            break
    server.global_state = _clone_state(best_global_state)
    if server._uses_oracle_evaluation() and best_oracle_state is not None:
        server.oracle_global_state = _clone_state(best_oracle_state)
    logger.info("Restored best federated checkpoint from round {} for final test", best_round)
    test_metrics = server.test_global()
    protocol_test_metrics, oracle_test_metrics = _resolve_test_metric_views(server, test_metrics)
    final_test_step = max(len(server.history), best_round + 1)
    attack_manager.finalize()
    total_elapsed = time.perf_counter() - start_time
    server.save(output_dir, config)
    final_log_payload = {**{f"test/{key}": value for key, value in test_metrics.items()}, "run/total_time_seconds": total_elapsed, "run/evaluation_mode": server.evaluation_mode}
    if protocol_test_metrics is not None:
        final_log_payload.update({f"protocol_test/{key}": value for key, value in protocol_test_metrics.items()})
    if oracle_test_metrics is not None:
        final_log_payload.update({f"oracle_test/{key}": value for key, value in oracle_test_metrics.items()})
    tracker.log(final_log_payload)
    try:
        _log_prediction_views(
            tracker,
            "prediction/federated/test_protocol",
            "federated test protocol prediction",
            server.model,
            test_loader,
            device,
            step=final_test_step,
            client_ids=client_ids,
            state=server.global_state,
        )
        oracle_state = server.oracle_global_state if server.oracle_global_state is not None else server.global_state
        _log_prediction_views(
            tracker,
            "prediction/federated/test_oracle",
            "federated test oracle prediction",
            server.model,
            test_loader,
            device,
            step=final_test_step,
            client_ids=client_ids,
            state=oracle_state,
        )
    except Exception as exc:
        logger.debug("Skip federated prediction plot: {}", exc)
    attack_records = save_attack_artifacts(output_dir, attack_manager.attack_results)
    with (output_dir / "attack_results.json").open("w", encoding="utf-8") as handle:
        json.dump(attack_records, handle, ensure_ascii=False, indent=2)
    attack_summary = summarize_attack_results(
        attack_manager.attack_results,
        float(config.get("attack", {}).get("success_rate_threshold", 0.03)),
    )
    summary = _build_federated_summary(
        server=server,
        test_metrics=test_metrics,
        total_elapsed=total_elapsed,
        best_round=best_round,
        best_metrics=best_metrics,
        attack_records=attack_records,
        attack_summary=attack_summary,
        attack_target_type=attack_target_type,
        protocol_test_metrics=protocol_test_metrics,
        oracle_test_metrics=oracle_test_metrics,
    )
    tracker.log({
        "run/best_round": best_round,
        "run/best_val_mse": best_metrics["mse"],
        "run/best_val_mae": best_metrics["mae"],
        "run/best_val_mape": best_metrics["mape"],
        "privacy/epsilon": summary["privacy_epsilon"],
        "privacy/delta": summary["privacy_delta"],
        "privacy/rdp_total": summary["privacy_rdp_total"],
        "privacy/sampling_rate": summary["privacy_sampling_rate"],
        "privacy/adaptive_clip_norm": summary["adaptive_clip_norm"],
    })
    tracker.finish()
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("Finished experiment: {}", summary)
    return summary
