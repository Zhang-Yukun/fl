"""High-level training algorithms: centralized, FedAvg, and compressed FedAvg."""

from __future__ import annotations

import copy
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
from fedlab.security.attack_common import (
    apply_set_recovery_metrics,
    attack_success_rate,
    attach_attack_metadata,
)
from fedlab.security.registry import (
    compute_recovery_metric_matrix,
    list_registered_attack_tracking_metrics,
    resolve_recovery_objective,
    run_attacks,
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
from fedlab.tasks import primary_metric as task_primary_metric, primary_metric_mode as task_primary_metric_mode
from fedlab.federated.protocol import validate_transport_modes
from fedlab.utils.tracking import Tracker
from fedlab.engine.training import build_training_optimizer, evaluate, predict_first_batch, predict_first_batch_for_state, train_one_epoch
from fedlab.utils.random import seed_cuda_device


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


def setup_seed(seed: int, deterministic: bool = True, *, device: torch.device | None = None) -> None:
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
    seed_cuda_device(seed_value, device)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception as exc:  # pragma: no cover - depends on torch build/runtime
        logger.warning("Could not set deterministic torch algorithms: {}", exc)
    logger.info("Set runtime seed={} deterministic={} device={}", seed_value, deterministic, device or "cpu")


def configure_random_seed(config: dict[str, Any], *, device: torch.device | None = None) -> None:
    """Apply the configured runtime seed when one is provided."""

    runtime_cfg = config.get("runtime", {})
    seed = runtime_cfg.get("seed")
    if seed is None:
        return
    resolved_device = resolve_device(config) if device is None else device
    setup_seed(int(seed), deterministic=bool(runtime_cfg.get("deterministic", True)), device=resolved_device)


def _resolve_training_epochs(config: dict[str, Any]) -> int:
    """Resolve the shared epoch budget used by centralized and local training."""

    return int(config.get("training", {}).get("epochs", 1))


def _log_mode_specific_schedule(config: dict[str, Any], mode: str) -> None:
    """Log when a mode is carrying schedule keys that do not apply to it."""

    if mode == "centralized":
        rounds = config.get("federated", {}).get("rounds")
        if rounds is not None:
            logger.info("Centralized mode ignores federated.rounds={} and uses training.epochs", rounds)


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
    val_metrics = data.pop('val_metrics', {})
    protocol_val_metrics = data.pop('protocol_val_metrics', {})
    payload = {f"round/{key}": value for key, value in data.items() if value is not None}
    payload.update({f"round/val_{key}": value for key, value in val_metrics.items()})
    payload.update({f"round/protocol_val_{key}": value for key, value in protocol_val_metrics.items()})
    for client in clients:
        prefix = f"client/{client['client_id']}"
        for key, value in client.items():
            if key != "client_id":
                payload[f"{prefix}/{key}"] = value
    return payload


def _configured_primary_metric_name(config: dict[str, Any]) -> str:
    """Return the task-specific metric used for checkpointing and early stopping."""

    return task_primary_metric(config)


def _configured_primary_metric_mode(config: dict[str, Any]) -> str:
    """Return whether the task-specific primary metric should be minimized or maximized."""

    return task_primary_metric_mode(config)


def _metric_series_payload(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Flatten one metric dictionary into ``prefix_metric`` scalar entries."""

    return {f"{prefix}_{name}": float(value) for name, value in metrics.items()}


def _augment_best_metric_summary(payload: dict[str, Any], best_metrics: dict[str, float], primary_metric_name: str) -> dict[str, Any]:
    """Attach generic best-validation metric fields to one summary payload."""

    payload['best_val'] = {key: float(value) for key, value in best_metrics.items()}
    payload['best_val_metric_name'] = primary_metric_name
    payload['best_val_metric_value'] = float(best_metrics[primary_metric_name])
    for name, value in best_metrics.items():
        payload[f'best_val_{name}'] = float(value)
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
    protocol_test_metrics: dict[str, float] | None = None,
    transport: str | None = None,
) -> dict[str, Any]:
    """Build a summary payload shared by final outputs and periodic snapshots."""

    history = server.history
    last_privacy = history[-1] if history else None
    protocol_test = test_metrics if protocol_test_metrics is None else protocol_test_metrics
    primary_metric_name = _configured_primary_metric_name(server.config)
    summary = {
        "test": test_metrics,
        "protocol_test": protocol_test,
        "rounds": len(history),
        "total_time_seconds": total_elapsed,
        "best_round": best_round,
        "test_checkpoint": "best_validation",
        "last_parameter_download_compression_ratio": history[-1].parameter_download_compression_ratio if history else 0.0,
        "last_parameter_upload_compression_ratio": history[-1].parameter_upload_compression_ratio if history else 0.0,
        "last_parameter_total_communication_ratio": history[-1].parameter_total_communication_ratio if history else 0.0,
        **_round_history_communication_summary(history),
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
    _augment_best_metric_summary(summary, best_metrics, primary_metric_name)
    if transport is not None:
        summary["transport"] = transport
    return summary


def _build_federated_resume_state(
    *,
    round_index: int,
    server: FederatedServer,
    best_global_state: StateDict,
    best_metrics: dict[str, float],
    best_round: int,
) -> dict[str, Any]:
    """Capture the minimal federated state needed to continue a stopped run."""

    return {
        "round_index": int(round_index),
        "global_state": _clone_state(server.global_state),
        "best_global_state": _clone_state(best_global_state),
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
    best_metrics: dict[str, float],
    best_round: int,
    transport: str | None = None,
) -> None:
    """Persist a periodic round snapshot without waiting for unfinished async attacks."""

    if not should_save_periodic_artifacts(config, round_index + 1):
        return
    test_metrics = server.test_global()
    protocol_test_metrics = test_metrics
    attack_records: list[dict[str, Any]] = []
    summary = _build_federated_summary(
        server=server,
        test_metrics=test_metrics,
        total_elapsed=time.perf_counter() - start_time,
        best_round=best_round,
        best_metrics=best_metrics,
        protocol_test_metrics=protocol_test_metrics,
        transport=transport,
    )
    snapshot_dir = save_federated_snapshot(
        output_dir,
        config,
        snapshot_name=f"round_{round_index + 1:04d}",
        model_state=server.global_state,
        metrics_history=server.history,
        summary=summary,
        attack_records=attack_records,
        resume_state=_build_federated_resume_state(
            round_index=round_index + 1,
            server=server,
            best_global_state=best_global_state,
            best_metrics=best_metrics,
            best_round=best_round,
        ),
    )
    logger.info("Saved periodic snapshot at round {} to {}", round_index, snapshot_dir)


def _wandb_cumulative_communication_payload(history: list[RoundRecord]) -> dict[str, float | int]:
    """Return cumulative communication metrics in a wandb-friendly flat namespace."""

    summary = _round_history_communication_summary(history)
    return {f"cumulative/{key}": value for key, value in summary.items()}



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
    """Return the only supported interception target for reconstruction attacks."""

    configured = str(config.get("attack", {}).get("target_type", "update_payload")).lower()
    if configured not in {"", "update_payload"}:
        logger.warning("Ignoring unsupported attack.target_type=%s and using update_payload", configured)
    return "update_payload"


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
    sample_x_shape: tuple[int, ...]
    sample_y_shape: tuple[int, ...]
    sample_x_dtype: str
    sample_y_dtype: str
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


def _torch_dtype_name(dtype: torch.dtype) -> str:
    """Return one stable dtype name for saved attack templates."""

    return str(dtype).replace("torch.", "")


def _resolve_torch_dtype(name: str) -> torch.dtype:
    """Resolve one saved dtype name back into a torch dtype."""

    dtype = getattr(torch, str(name), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype name in saved attack template: {name}")
    return dtype


def _materialize_attack_template(shape: tuple[int, ...], dtype_name: str) -> torch.Tensor:
    """Create one zero-valued template tensor used only for shape and dtype."""

    return torch.zeros(shape, dtype=_resolve_torch_dtype(dtype_name))


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


def _client_attack_available_samples(client: FederatedClient) -> int | None:
    """Best-effort training-sample count used by attack sampling defaults."""

    getter = getattr(client, "train_num_samples", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            pass
    reference_getter = getattr(client, "train_reference_inputs", None)
    if callable(reference_getter):
        try:
            reference_inputs = reference_getter()
            return int(reference_inputs.shape[0])
        except Exception:
            pass
    train_loader = getattr(client, "train_loader", None)
    dataset = getattr(train_loader, "dataset", None) if train_loader is not None else None
    if dataset is None:
        return None
    try:
        return int(len(dataset))
    except Exception:
        return None


def _resolve_attack_max_samples(attack_cfg: dict[str, Any], available_samples: int | None) -> int | None:
    """Resolve how many samples each attack reconstruction should include."""

    configured = attack_cfg.get("max_samples", 1)
    if configured is None:
        return None
    if str(configured).strip().lower() == "auto":
        count = None if available_samples is None else max(1, int(available_samples))
        cap = attack_cfg.get("max_samples_cap")
        if count is not None and cap is not None:
            count = min(count, max(1, int(cap)))
        return count
    return max(1, int(configured))


def _resolve_client_scale_metadata(client: FederatedClient) -> tuple[list[float] | None, list[float] | None]:
    """Return scaler statistics for one client, preferring registered metadata when present."""

    registered_mean = getattr(client, 'registered_scale_mean', None)
    registered_std = getattr(client, 'registered_scale_std', None)
    if registered_mean is not None or registered_std is not None:
        return registered_mean, registered_std
    train_loader = getattr(client, 'train_loader', None)
    scaler = getattr(train_loader, 'scaler', None)
    scale_mean = None if getattr(scaler, 'mean', None) is None else [float(value) for value in scaler.mean.reshape(-1).tolist()]
    scale_std = None if getattr(scaler, 'std', None) is None else [float(value) for value in scaler.std.reshape(-1).tolist()]
    return scale_mean, scale_std


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
        scale_mean, scale_std = _resolve_client_scale_metadata(client)
        reference_inputs = client.train_reference_inputs()
        reference_target_getter = getattr(client, "train_reference_targets", None)
        reference_targets = reference_target_getter() if callable(reference_target_getter) else None
        samples = []
        max_samples = _resolve_attack_max_samples(attack_cfg, _client_attack_available_samples(client))
        sample_x, sample_y = client.sample_batch(max_samples=max_samples, batch_index=0)
        samples.append(
            {
                "sample_index": 0,
                "sample_x_shape": list(sample_x.shape),
                "sample_y_shape": list(sample_y.shape),
                "sample_x_dtype": _torch_dtype_name(sample_x.dtype),
                "sample_y_dtype": _torch_dtype_name(sample_y.dtype),
            }
        )
        records.append(
            {
                "client_id": client.client_id,
                "round_index": int(round_index),
                "target_type": "update_payload",
                "aggregation_payload_kind": getattr(result, "aggregation_payload_kind", "dense_update"),
                "compressor": getattr(result, "compressor", "none"),
                "round_base_state": _clone_state(round_base_state),
                "target_update": _clone_state(target_update),
                "reference_inputs": None if reference_inputs is None else reference_inputs.detach().cpu().clone(),
                "reference_targets": None if reference_targets is None else reference_targets.detach().cpu().clone(),
                "scale_mean": scale_mean,
                "scale_std": scale_std,
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
                    sample_x_shape=tuple(int(value) for value in sample["sample_x_shape"]),
                    sample_y_shape=tuple(int(value) for value in sample["sample_y_shape"]),
                    sample_x_dtype=str(sample["sample_x_dtype"]),
                    sample_y_dtype=str(sample["sample_y_dtype"]),
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
    sample_lookup = {(sample.client_id, sample.round_index, sample.sample_index): sample for sample in task.samples}
    for sample in task.samples:
        sample_x = _materialize_attack_template(sample.sample_x_shape, sample.sample_x_dtype)
        sample_y = _materialize_attack_template(sample.sample_y_shape, sample.sample_y_dtype)
        for result in run_attacks(
            config,
            sample.round_base_state,
            sample.target,
            sample_x,
            sample_y,
            attack_device,
            target_type=sample.target_type,
            reference_inputs=sample.reference_inputs,
            reference_targets=sample.reference_targets,
        ):
            result = attach_attack_metadata(
                result,
                client_id=sample.client_id,
                round_index=sample.round_index,
                sample_index=sample.sample_index,
            )
            plot_reference_x = getattr(result, "reference_x", None)
            plot_reference_y = getattr(result, "reference_y", None)
            result.plot_reference_x = None if plot_reference_x is None else _inverse_plot_tensor(plot_reference_x, sample.scale_mean, sample.scale_std)
            result.plot_reconstructed_x = _inverse_plot_tensor(result.reconstructed_x, sample.scale_mean, sample.scale_std)
            result.plot_reference_y = None if plot_reference_y is None else _inverse_plot_tensor(plot_reference_y, sample.scale_mean, sample.scale_std)
            result.plot_reconstructed_y = None if result.reconstructed_y is None else _inverse_plot_tensor(result.reconstructed_y, sample.scale_mean, sample.scale_std)
            attacks.append(result)
    grouped: dict[tuple[str | None, int | None, str], list[Any]] = {}
    for result in attacks:
        grouped.setdefault((result.client_id, result.round_index, result.name), []).append(result)
    for key, subset in grouped.items():
        client_id, round_index, _name = key
        first = next((sample_lookup.get((result.client_id, result.round_index, result.sample_index)) for result in subset), None)
        if first is None or first.reference_inputs is None:
            continue
        apply_set_recovery_metrics(
            subset,
            reference_inputs=first.reference_inputs,
            reference_targets=first.reference_targets,
            config=config,
        )
        for result in subset:
            if result.reference_x is not None:
                result.plot_reference_x = _inverse_plot_tensor(result.reference_x, first.scale_mean, first.scale_std)
            if result.reference_y is not None:
                result.plot_reference_y = _inverse_plot_tensor(result.reference_y, first.scale_mean, first.scale_std)
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
    payload[f"{prefix}/primary_metric_name"] = getattr(subset[0], "metric_name", "budget_recovered_fraction")
    payload[f"{prefix}/primary_metric_value"] = sum(result.mse for result in subset) / len(subset)
    payload[f"{prefix}/cumulative_avg_primary_metric_value"] = 0.0 if not cumulative_subset else sum(result.mse for result in cumulative_subset) / len(cumulative_subset)
    for spec in list_registered_attack_tracking_metrics():
        values = [spec.value_getter(result) for result in subset if spec.value_getter(result) is not None]
        if values:
            payload[f"{prefix}/{spec.current_key}"] = _mean_finite(values)
        if spec.cumulative_key is not None:
            cumulative_values = [spec.value_getter(result) for result in cumulative_subset if spec.value_getter(result) is not None]
            if cumulative_values:
                payload[f"{prefix}/{spec.cumulative_key}"] = _mean_finite(cumulative_values)
    payload[f"{prefix}/success_fraction"] = sum(float(result.success) for result in subset) / len(subset)
    payload[f"{prefix}/cumulative_success_rate"] = attack_success_rate(cumulative_subset)
    return payload


def _round_attack_payload(round_result: AttackRoundResult, cumulative_results: list[Any]) -> dict[str, float | str]:
    """Build per-round and cumulative attack metrics for tracking/logging."""

    round_attacks = round_result.attacks
    primary_metric_name = "budget_recovered_fraction" if not round_attacks else getattr(round_attacks[0], "metric_name", "budget_recovered_fraction")
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


def _attack_plot_data_range(config: dict[str, Any]) -> float:
    """Return the positive data range used by recovery metric helpers."""

    value = float(config.get("attack", {}).get("data_range", 1.0))
    return value if value > 0 else 1.0


def _attack_result_batch_size(result: Any) -> int | None:
    """Return the batch size for one attack result payload when available."""

    for field in ("reconstructed_x", "plot_reconstructed_x", "reference_x", "plot_reference_x"):
        tensor = getattr(result, field, None)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return None


def _slice_attack_tensor(tensor: Any, row_index: int, batch_size: int | None) -> Any:
    """Return one batch row for visualization-only attack payloads."""

    if not isinstance(tensor, torch.Tensor) or batch_size is None or batch_size <= 1 or tensor.ndim == 0:
        return tensor
    if int(tensor.shape[0]) != batch_size:
        return tensor
    row = max(0, min(int(row_index), batch_size - 1))
    return tensor[row:row + 1].detach().cpu().clone()


def _attack_result_row_metrics(result: Any, config: dict[str, Any]) -> tuple[list[float] | None, str | None]:
    """Return one per-row matched metric list for a result when available."""

    metric_name = getattr(result, "matched_reference_metric_name", None)
    reconstructed_x = getattr(result, "reconstructed_x", None)
    reference_x = getattr(result, "reference_x", None)
    if metric_name is None or not isinstance(reconstructed_x, torch.Tensor) or not isinstance(reference_x, torch.Tensor):
        return None, None
    if reconstructed_x.ndim == 0 or reference_x.ndim == 0:
        return None, None
    if reconstructed_x.shape[0] == 0 or reconstructed_x.shape[0] != reference_x.shape[0]:
        return None, None
    metric_matrix = compute_recovery_metric_matrix(
        reconstructed_x.detach().cpu(),
        reference_x.detach().cpu(),
        str(metric_name),
        _attack_plot_data_range(config),
    )
    diagonal = metric_matrix.diagonal()
    return [float(value) for value in diagonal.tolist()], resolve_recovery_objective(None, str(metric_name))


def _select_attack_visualization_result(results: list[Any], config: dict[str, Any]) -> Any:
    """Choose one best-matched sample and return a sliced visualization payload."""

    if not results:
        return None
    best_result = results[0]
    best_row_index = 0
    best_score: float | None = None
    best_objective: str | None = None
    for result in results:
        row_metrics, objective = _attack_result_row_metrics(result, config)
        if not row_metrics or objective not in {"min", "max"}:
            continue
        candidate_score = min(row_metrics) if objective == "min" else max(row_metrics)
        candidate_row_index = row_metrics.index(candidate_score)
        if best_score is None:
            best_result = result
            best_row_index = candidate_row_index
            best_score = float(candidate_score)
            best_objective = objective
            continue
        if best_objective != objective:
            continue
        is_better = candidate_score < best_score if objective == "min" else candidate_score > best_score
        if is_better:
            best_result = result
            best_row_index = candidate_row_index
            best_score = float(candidate_score)
            best_objective = objective
    visualization_result = copy.copy(best_result)
    batch_size = _attack_result_batch_size(best_result)
    for field in (
        "reference_x",
        "reference_y",
        "reconstructed_x",
        "reconstructed_y",
        "plot_reference_x",
        "plot_reference_y",
        "plot_reconstructed_x",
        "plot_reconstructed_y",
    ):
        setattr(
            visualization_result,
            field,
            _slice_attack_tensor(getattr(best_result, field, None), best_row_index, batch_size),
        )
    matched_indices = getattr(best_result, "matched_reference_indices", None)
    if matched_indices is not None:
        if 0 <= best_row_index < len(matched_indices):
            visualization_result.matched_reference_indices = [int(matched_indices[best_row_index])]
        else:
            visualization_result.matched_reference_indices = None
    if best_score is not None:
        visualization_result.matched_reference_metric_value = float(best_score)
        visualization_result.matched_reference_metric_min_value = float(best_score)
    return visualization_result


def _log_attack_reconstruction_views(tracker: Tracker, results: list[Any], step: int, config: dict[str, Any]) -> None:
    """Log one best-match reconstruction figure per method and per client."""

    if not hasattr(tracker, "log_attack_reconstruction"):
        return
    method_groups: dict[str, list[Any]] = {}
    client_method_groups: dict[tuple[str, str], list[Any]] = {}
    for result in results:
        method = str(result.name)
        method_groups.setdefault(method, []).append(result)
        client_id = getattr(result, "client_id", None)
        if client_id is not None:
            client_method_groups.setdefault((str(client_id), method), []).append(result)
    for method, subset in method_groups.items():
        visualization_result = _select_attack_visualization_result(subset, config)
        if visualization_result is None:
            continue
        tracker.log_attack_reconstruction(f"attack/{method}/reconstruction", visualization_result, step=step)
    for (client_id, method), subset in client_method_groups.items():
        visualization_result = _select_attack_visualization_result(subset, config)
        if visualization_result is None:
            continue
        tracker.log_attack_reconstruction(
            f"attack/client/{client_id}/{method}/reconstruction",
            visualization_result,
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
            _log_attack_reconstruction_views(self.tracker, round_result.attacks, round_index, self.config)
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
    device = resolve_device(config)
    configure_random_seed(config, device=device)
    _log_mode_specific_schedule(config, "centralized")
    tracker = Tracker(config)
    start_time = time.perf_counter()
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
    primary_metric_name = _configured_primary_metric_name(config)
    primary_metric_mode = _configured_primary_metric_mode(config)
    stopper = EarlyStopper(int(config["training"].get("patience", 5)), float(config["training"].get("min_delta", 0.0)), mode=primary_metric_mode)
    history = []
    best_state = serialize_model(model)
    best_metrics: dict[str, float] | None = None
    best_round = -1
    for round_index in range(_resolve_training_epochs(config)):
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
            metric_name=primary_metric_name,
            metric_mode=primary_metric_mode,
        )
        round_time = time.perf_counter() - round_start
        elapsed = time.perf_counter() - start_time
        round_record = {
            "round": round_index,
            "train_loss": loss,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": metrics[primary_metric_name],
            **_metric_series_payload("val", metrics),
            "round_time_seconds": round_time,
            "elapsed_time_seconds": elapsed,
        }
        history.append(round_record)
        logger.info(
            "Centralized round {} loss={:.6f} val_{}={:.6f} time={:.2f}s elapsed={:.2f}s",
            round_index,
            loss,
            primary_metric_name,
            metrics[primary_metric_name],
            round_time,
            elapsed,
        )
        tracker.log({
            "round/loss": loss,
            "round/val_primary_metric_name": primary_metric_name,
            "round/val_primary_metric_value": metrics[primary_metric_name],
            **{f"round/val_{key}": value for key, value in metrics.items()},
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
        if stopper.update(metrics[primary_metric_name]):
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
                "best_val_metric_name": primary_metric_name,
                "best_val_metric_value": best_metrics[primary_metric_name],
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
        "run/best_val_metric_name": primary_metric_name,
        "run/best_val_metric_value": best_metrics[primary_metric_name],
        **{f"run/best_val_{key}": value for key, value in best_metrics.items()},
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
        summary_payload = {
            "test": test_metrics,
            "rounds": len(history),
            "total_time_seconds": total_elapsed,
            "best_round": best_round,
            "best_val": best_metrics,
            "best_val_metric_name": primary_metric_name,
            "best_val_metric_value": best_metrics[primary_metric_name],
            "test_checkpoint": "best_validation",
        }
        for key, value in best_metrics.items():
            summary_payload[f"best_val_{key}"] = value
        json.dump(
            summary_payload,
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
    device = resolve_device(config)
    configure_random_seed(config, device=device)
    validate_transport_modes(config)
    _log_mode_specific_schedule(config, "federated")
    tracker = Tracker(config)
    start_time = time.perf_counter()
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    client_ids = list(train_loaders.keys())
    server = FederatedServer(config, val_loader, test_loader, device)
    total_train_samples = sum(_loader_num_samples(loader) for loader in train_loaders.values())
    server.total_clients = len(train_loaders)
    server.total_train_samples = total_train_samples
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
    primary_metric_name = _configured_primary_metric_name(config)
    primary_metric_mode = _configured_primary_metric_mode(config)
    stopper = EarlyStopper(int(config["training"].get("patience", 5)), float(config["training"].get("min_delta", 0.0)), mode=primary_metric_mode)
    max_rounds = int(config["federated"].get("rounds", 20))
    best_global_state = _clone_state(server.global_state)
    best_metrics: dict[str, float] | None = None
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
        protocol_metrics = metrics
        best_global_state, best_metrics, best_round, improved = _update_best_checkpoint(
            best_state=best_global_state,
            best_metrics=best_metrics,
            best_index=best_round,
            candidate_state=server.global_state,
            candidate_metrics=metrics,
            candidate_index=round_index,
            label="round",
            metric_name=primary_metric_name,
            metric_mode=primary_metric_mode,
        )
        record = server.record_round(
            round_index,
            results,
            aggregation_weights,
            metrics,
            round_time_seconds=time.perf_counter() - round_start,
            elapsed_time_seconds=time.perf_counter() - start_time,
            protocol_metrics=protocol_metrics,
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
        _save_periodic_federated_snapshot(
            output_dir=output_dir,
            config=config,
            server=server,
            round_index=round_index,
            start_time=start_time,
            best_global_state=best_global_state,
            best_metrics=best_metrics,
            best_round=best_round,
        )
        if stopper.update(metrics[primary_metric_name]):
            logger.info("Early stopping at round {}", round_index)
            break
    server.global_state = _clone_state(best_global_state)
    logger.info("Restored best federated checkpoint from round {} for final test", best_round)
    test_metrics = server.test_global()
    protocol_test_metrics = test_metrics
    final_test_step = max(len(server.history), best_round + 1)
    total_elapsed = time.perf_counter() - start_time
    server.save(output_dir, config)
    final_log_payload = {**{f"test/{key}": value for key, value in test_metrics.items()}, "run/total_time_seconds": total_elapsed}
    if protocol_test_metrics is not None:
        final_log_payload.update({f"protocol_test/{key}": value for key, value in protocol_test_metrics.items()})
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
    except Exception as exc:
        logger.debug("Skip federated prediction plot: {}", exc)
    summary = _build_federated_summary(
        server=server,
        test_metrics=test_metrics,
        total_elapsed=total_elapsed,
        best_round=best_round,
        best_metrics=best_metrics,
        protocol_test_metrics=protocol_test_metrics,
    )
    tracker.log({
        "run/best_round": best_round,
        "run/best_val_metric_name": primary_metric_name,
        "run/best_val_metric_value": best_metrics[primary_metric_name],
        **{f"run/best_val_{key}": value for key, value in best_metrics.items()},
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
