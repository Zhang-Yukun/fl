"""High-level training algorithms: centralized, FedAvg, and compressed FedAvg."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from fedlab.utils.artifacts import save_experiment_config, save_federated_snapshot, should_save_periodic_artifacts
from fedlab.replay_capture.artifacts import save_captured_update_records
from fedlab.federated.client import FederatedClient
from fedlab.federated.methods import build_method, is_registered_compressed
from fedlab.datasets import build_federated_loaders
from fedlab.utils.logging import setup_logging
from fedlab.utils.runtime import configure_random_seed, configure_torch_runtime, resolve_device
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


def _resolve_training_epochs(config: dict[str, Any]) -> int:
    """Resolve the shared epoch budget used by centralized and local training."""

    return int(config.get("training", {}).get("epochs", 1))


def _log_mode_specific_schedule(config: dict[str, Any], mode: str) -> None:
    """Log when a mode is carrying schedule keys that do not apply to it."""

    if mode == "centralized":
        rounds = config.get("federated", {}).get("rounds")
        if rounds is not None:
            logger.info("Centralized mode ignores federated.rounds={} and uses training.epochs", rounds)


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

    capture_cfg = config.get("replay_capture", {})
    if not capture_cfg.get("enabled", True):
        return False
    frequency = int(capture_cfg.get("frequency_rounds", 30))
    return round_index == 0 or round_index == max_rounds - 1 or (frequency > 0 and round_index % frequency == 0)


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



def _torch_dtype_name(dtype: torch.dtype) -> str:
    """Return one stable dtype name for saved attack templates."""

    return str(dtype).replace("torch.", "")


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


def _resolve_capture_max_samples(capture_cfg: dict[str, Any], available_samples: int | None) -> int | None:
    """Resolve how many training samples each saved replay template should include."""

    configured = capture_cfg.get("max_samples", 1)
    if configured is None:
        return None
    if str(configured).strip().lower() == "auto":
        count = None if available_samples is None else max(1, int(available_samples))
        cap = capture_cfg.get("max_samples_cap")
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
    capture_cfg = config.get("replay_capture", {})
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
        samples = []
        max_samples = _resolve_capture_max_samples(capture_cfg, _client_attack_available_samples(client))
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
                "scale_mean": scale_mean,
                "scale_std": scale_std,
                "samples": samples,
            }
        )
    return records




def _mean_finite(values: list[float]) -> float | None:
    """Return the mean of finite values, or None when all values are non-finite."""

    finite = [value for value in values if value is not None and isinstance(value, (int, float)) and float("-inf") < float(value) < float("inf")]
    if not values:
        return 0.0
    if not finite:
        return None
    return sum(float(value) for value in finite) / len(finite)


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
