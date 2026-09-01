"""Multi-process federated training built on the generic gRPC transport.

Example:
    Server:
        ``python -m fedlab.entrypoints.server --config configs/rare/fedavg.yaml``

    Client:
        ``python -m fedlab.entrypoints.client --client-id Nd2O3 --config configs/rare/fedavg.yaml``
"""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from fedlab.communication.grpc_service import FederatedRpcClient, FederatedRpcServer
from fedlab.datasets import build_federated_loaders, build_server_evaluation_loaders
from fedlab.datasets.image_classification import build_client_image_classification_train_loader
from fedlab.datasets.rare_earth import build_client_rare_earth_train_loader
from fedlab.federated.algorithms import (
    _build_federated_resume_state,
    _build_federated_summary,
    _capture_round_update_records,
    _clone_state,
    _round_history_communication_summary,
    _save_periodic_federated_snapshot,
    _update_best_checkpoint,
    _wandb_cumulative_communication_payload,
    _wandb_round_payload,
    _log_prediction_views,
    _configured_primary_metric_name,
    _configured_primary_metric_mode,
    is_compressed_algorithm,
)
from fedlab.federated.client import ClientResult, FederatedClient
from fedlab.federated.server import EarlyStopper, FederatedServer
from fedlab.federated.protocol import validate_transport_modes
from fedlab.replay_capture.artifacts import save_captured_update_records
from fedlab.tasks.registry import task_type
from fedlab.utils.logging import setup_logging
from fedlab.utils.runtime import configure_random_seed, configure_torch_runtime, resolve_device
from fedlab.utils.artifacts import save_experiment_config, should_save_periodic_artifacts
from fedlab.utils.serialization import state_num_bytes, state_num_parameters
from fedlab.utils.transport import estimate_upload_transport_bytes
from fedlab.utils.tracking import Tracker
from fedlab.engine.training import predict_first_batch, predict_first_batch_for_state


def _format_num_bytes(num_bytes: int) -> str:
    """Format raw bytes into a compact human-readable string."""

    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def _transport_delta(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    """Return the serialized transport delta between two RPC counter snapshots."""

    return {key: int(current.get(key, 0)) - int(previous.get(key, 0)) for key in current.keys()}



def _loader_num_samples(loader: Any) -> int:
    """Return the number of samples carried by one loader-like object."""

    dataset = getattr(loader, "dataset", None)
    return len(dataset) if dataset is not None else len(loader)




def _build_grpc_client_training_state(config: dict[str, Any], client_id: str) -> tuple[Any, int]:
    """Build one client's local training loader plus its local sample count."""

    resolved_task_type = task_type(config)
    if resolved_task_type == 'classification' and 'split_dir' in config.get('data', {}):
        loader = build_client_image_classification_train_loader(config, client_id)
        return loader, _loader_num_samples(loader)
    if resolved_task_type == 'forecasting':
        loader = build_client_rare_earth_train_loader(config, client_id)
        return loader, _loader_num_samples(loader)
    train_loaders, _, _ = build_federated_loaders(config)
    if client_id not in train_loaders:
        raise ValueError(f'Unknown client_id {client_id}; expected one of {sorted(train_loaders)}')
    loader = train_loaders[client_id]
    return loader, _loader_num_samples(loader)

def _apply_transport_metrics(
    result: ClientResult,
    transport_delta: dict[str, int],
    round_index: int | None = None,
) -> ClientResult:
    """Attach serialized transport counters to one client result.

    Example:
        ``result = _apply_transport_metrics(result, {"sent_bytes": 128, "received_bytes": 64}, round_index=0)``
        records the actual serialized RPC bytes for that round.
    """

    result.transport_download_bytes = max(0, int(transport_delta.get('received_bytes', 0)))
    result.transport_download_overhead_bytes = max(0, result.transport_download_bytes - result.parameter_download_bytes)
    base_upload_bytes = max(0, int(transport_delta.get('sent_bytes', 0)))
    if round_index is not None:
        estimate_upload_transport_bytes(result, round_index=round_index, base_bytes=base_upload_bytes)
    else:
        result.transport_upload_bytes = base_upload_bytes
        result.transport_upload_overhead_bytes = max(0, result.transport_upload_bytes - result.parameter_upload_bytes)
    return result


class GrpcFederatedCoordinator:
    """Server-side coordinator that aggregates updates when all clients report."""

    def __init__(self, config: dict[str, Any]):
        """Initialize server-side state for multi-process training."""

        self.config = config
        self.output_dir = Path(config['experiment']['output_dir'])
        setup_logging(self.output_dir, config.get('runtime', {}).get('log_level', 'INFO'))
        config_formats = config.get('artifacts', {}).get('config_formats')
        saved_configs = save_experiment_config(config, self.output_dir, config_formats)
        logger.info('Saved startup config artifacts: {}', [str(path) for path in saved_configs])
        configure_torch_runtime(config)
        self.device = resolve_device(config)
        configure_random_seed(config, device=self.device)
        validate_transport_modes(config)
        self.tracker = Tracker(config)
        configured_clients = tuple(config.get('data', {}).get('clients') or ())
        if configured_clients:
            self.expected_clients = configured_clients
        else:
            train_loaders, _, _ = build_federated_loaders(config)
            self.expected_clients = tuple(train_loaders.keys())
        val_loader, test_loader = build_server_evaluation_loaders(config)
        self.server = FederatedServer(config, val_loader, test_loader, self.device)
        self.server.total_clients = len(self.expected_clients)
        self.server.total_train_samples = None
        self.registered_client_samples: dict[str, int] = {}
        self.registered_client_metadata: dict[str, dict[str, Any]] = {}
        self.registration_ready = False
        self.evaluation_ready = val_loader is not None and test_loader is not None
        self.capture_client_ids = list(self.expected_clients)
        self.compressed = is_compressed_algorithm(config)
        self.max_rounds = int(config['federated'].get('rounds', 20))
        self.primary_metric_name = _configured_primary_metric_name(config)
        self.primary_metric_mode = _configured_primary_metric_mode(config)
        self.stopper = EarlyStopper(
            int(config['training'].get('patience', 5)),
            float(config['training'].get('min_delta', 0.0)),
            mode=self.primary_metric_mode,
        )
        self.round_index = 0
        self.start_time = time.perf_counter()
        self.round_start_time = time.perf_counter()
        self.pending: dict[str, Any] = {}
        self.stopped = False
        self.stop_acked_clients: set[str] = set()
        self.stop_time_seconds: float | None = None
        self.finalize_requested = False
        self.finalization_started = False
        self.finalization_completed = False
        self.pending_final_round_thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.best_global_state = _clone_state(self.server.global_state)
        self.best_metrics: dict[str, float] | None = None
        self.best_round = -1
        self.runtime_ready = self.server.runtime_ready()
        self.runtime_error: str | None = None
        self.runtime_init_thread: threading.Thread | None = None
        self.tracker.log({
            'run/algorithm': str(config['federated'].get('algorithm', 'fedavg')),
            'run/client_count': len(self.expected_clients),
            'run/model_parameters': state_num_parameters(self.server.global_state),
            'run/model_bytes': state_num_bytes(self.server.global_state),
            'run/compressed_uploads': self.compressed,
            'run/transport': 'grpc',
        })

    def start_runtime_initialization(self) -> None:
        """Kick off any deferred server runtime initialization after the gRPC server starts listening."""

        with self.lock:
            if self.runtime_ready or self.runtime_error is not None or self.runtime_init_thread is not None:
                return
            self.runtime_init_thread = threading.Thread(
                target=self._initialize_runtime,
                name='grpc-runtime-init',
                daemon=True,
            )
            self.runtime_init_thread.start()

    def _initialize_runtime(self) -> None:
        """Prepare any deferred algorithm-specific server runtime state."""

        try:
            self.server.initialize_runtime()
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception('Failed to initialize deferred gRPC server runtime: {}', exc)
            with self.lock:
                self.runtime_error = str(exc)
                self.stopped = True
                self.stop_time_seconds = time.perf_counter()
                self.finalization_completed = True
            return
        with self.lock:
            self.runtime_ready = True
        logger.info('Deferred gRPC server runtime initialization finished')

    def register_client(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Register one client and its local training-sample count before round-0."""

        client_id = str(payload.get('client_id', ''))
        local_train_samples = int(payload.get('local_train_samples', 0) or 0)
        scale_mean = payload.get('scale_mean')
        scale_std = payload.get('scale_std')
        with self.lock:
            if client_id not in self.expected_clients:
                return {
                    'accepted': False,
                    'error': f'Unknown client_id {client_id}',
                    'registered_clients': len(self.registered_client_samples),
                    'expected_clients': len(self.expected_clients),
                    'registration_ready': self.registration_ready,
                    'stop': self.stopped,
                }
            previous = self.registered_client_samples.get(client_id)
            self.registered_client_samples[client_id] = local_train_samples
            self.registered_client_metadata[client_id] = {
                'local_train_samples': local_train_samples,
                'scale_mean': None if scale_mean is None else [float(value) for value in scale_mean],
                'scale_std': None if scale_std is None else [float(value) for value in scale_std],
            }
            self.registration_ready = set(self.expected_clients).issubset(self.registered_client_samples)
            if previous is None:
                logger.info(
                    'Registered client {} with local_train_samples={} ({}/{})',
                    client_id,
                    local_train_samples,
                    len(self.registered_client_samples),
                    len(self.expected_clients),
                )
            elif previous != local_train_samples:
                logger.warning('Updated client {} registration local_train_samples {} -> {}', client_id, previous, local_train_samples)
            if self.registration_ready:
                self.server.total_train_samples = sum(self.registered_client_samples[registered_id] for registered_id in self.expected_clients)
                if not self.evaluation_ready:
                    val_loader, test_loader = build_server_evaluation_loaders(
                        self.config,
                        registration_metadata=self.registered_client_metadata,
                    )
                    if val_loader is None or test_loader is None:
                        raise RuntimeError('Server evaluation loaders are not ready after client registration')
                    self.server.val_loader = val_loader
                    self.server.test_loader = test_loader
                    self.evaluation_ready = True
                logger.info(
                    'All clients registered for gRPC training: total_clients={} total_train_samples={} evaluation_ready={}',
                    self.server.total_clients,
                    self.server.total_train_samples,
                    self.evaluation_ready,
                )
            return {
                'accepted': True,
                'error': None,
                'registered_clients': len(self.registered_client_samples),
                'expected_clients': len(self.expected_clients),
                'registration_ready': self.registration_ready,
                'stop': self.stopped,
            }

    def ack_stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record that one client received the stop signal."""

        client_id = str(payload.get('client_id', ''))
        with self.lock:
            if client_id in self.expected_clients:
                self.stop_acked_clients.add(client_id)
                logger.info(
                    'Received stop ack from {} ({}/{})',
                    client_id,
                    len(self.stop_acked_clients),
                    len(self.expected_clients),
                )
            return {
                'stop': self.stopped,
                'round': self.round_index,
                'acked_clients': len(self.stop_acked_clients),
                'expected_clients': len(self.expected_clients),
            }

    def ready_for_shutdown(self) -> bool:
        """Return whether the server can safely terminate the gRPC service."""

        with self.lock:
            return self.stopped and set(self.expected_clients).issubset(self.stop_acked_clients)

    def finalize_if_requested(self) -> bool:
        """Run final artifact persistence once, outside the RPC hot path."""

        pending_thread = None
        with self.lock:
            if not self.finalize_requested or self.finalization_started:
                return self.finalization_completed
            self.finalization_started = True
            pending_thread = self.pending_final_round_thread
        if pending_thread is not None:
            pending_thread.join()
        self._finalize()
        with self.lock:
            self.finalization_completed = True
        return True

    def get_global(self) -> dict[str, Any]:
        """Return the current global payload for remote clients."""

        with self.lock:
            ready = self.runtime_ready and self.registration_ready and self.evaluation_ready
            return {
                'round': self.round_index,
                'state': self.server.global_state if ready else None,
                'compressed': self.compressed,
                'round_context': self.server.build_round_context() if ready else {},
                'stop': self.stopped,
                'ready': ready,
                'runtime_ready': self.runtime_ready,
                'registration_ready': self.registration_ready,
                'registered_clients': len(self.registered_client_samples),
                'expected_clients': len(self.expected_clients),
                'runtime_error': self.runtime_error,
            }

    def _finish_final_round_bookkeeping(self, round_index: int, results, aggregation_weights, round_base_state, round_context: dict[str, Any] | None = None) -> None:
        """Complete final-round validation and artifact bookkeeping off the submit hot path."""

        metrics = self.server.evaluate_global()
        protocol_metrics = metrics
        self.best_global_state, self.best_metrics, self.best_round, improved = _update_best_checkpoint(
            best_state=self.best_global_state,
            best_metrics=self.best_metrics,
            best_index=self.best_round,
            candidate_state=self.server.global_state,
            candidate_metrics=metrics,
            candidate_index=round_index,
            label='round',
            metric_name=self.primary_metric_name,
            metric_mode=self.primary_metric_mode,
        )
        record = self.server.record_round(
            round_index,
            results,
            aggregation_weights,
            metrics,
            round_time_seconds=time.perf_counter() - self.round_start_time,
            elapsed_time_seconds=time.perf_counter() - self.start_time,
            protocol_metrics=protocol_metrics,
            silent=True,
        )
        self.tracker.log({**_wandb_round_payload(record), **_wandb_cumulative_communication_payload(self.server.history)}, step=round_index)
        try:
            _log_prediction_views(
                self.tracker,
                'prediction/grpc/val_protocol',
                'grpc val protocol prediction',
                self.server.model,
                self.server.val_loader,
                self.server.device,
                step=round_index,
                client_ids=list(self.expected_clients),
                state=self.server.global_state,
            )
        except Exception as exc:
            logger.debug('Skip gRPC val prediction plot: {}', exc)
        captured_update_records = _capture_round_update_records(
            self.config,
            self.capture_client_ids,
            results,
            round_index,
            self.max_rounds,
            round_base_state,
            server=self.server,
            round_context=round_context,
        )
        save_captured_update_records(self.output_dir, captured_update_records)
        if should_save_periodic_artifacts(self.config, round_index + 1):
            _save_periodic_federated_snapshot(
                output_dir=self.output_dir,
                config=self.config,
                server=self.server,
                round_index=round_index,
                start_time=self.start_time,
                best_global_state=self.best_global_state,
                best_metrics=self.best_metrics,
                best_round=self.best_round,
                transport='grpc',
            )

    def _finalize(self) -> None:
        """Persist final model artifacts and a summary compatible with the main FL path."""

        self.server.global_state = _clone_state(self.best_global_state)
        logger.info('Restored best gRPC federated checkpoint from round {} for final test', self.best_round)
        test_metrics = self.server.test_global()
        protocol_test_metrics = test_metrics
        self.server.save(self.output_dir, self.config)
        final_test_step = max(len(self.server.history), self.best_round + 1)
        total_elapsed = time.perf_counter() - self.start_time
        summary = _build_federated_summary(
            server=self.server,
            test_metrics=test_metrics,
            total_elapsed=total_elapsed,
            best_round=self.best_round,
            best_metrics=self.best_metrics,
            protocol_test_metrics=protocol_test_metrics,
            transport='grpc',
        )
        final_log_payload = {
            **{f'test/{key}': value for key, value in test_metrics.items()},
            'run/rounds': len(self.server.history),
            'run/total_time_seconds': total_elapsed,
            'run/transport': 'grpc',
            'run/best_round': self.best_round,
            'run/best_val_metric_name': self.primary_metric_name,
            'run/best_val_metric_value': self.best_metrics[self.primary_metric_name],
            **{f'run/best_val_{key}': value for key, value in self.best_metrics.items()},
            'privacy/epsilon': summary['privacy_epsilon'],
            'privacy/delta': summary['privacy_delta'],
            'privacy/rdp_total': summary['privacy_rdp_total'],
            'privacy/sampling_rate': summary['privacy_sampling_rate'],
            'privacy/adaptive_clip_norm': summary['adaptive_clip_norm'],
        }
        if protocol_test_metrics is not None:
            final_log_payload.update({f'protocol_test/{key}': value for key, value in protocol_test_metrics.items()})
        self.tracker.log(final_log_payload)
        try:
            _log_prediction_views(
                self.tracker,
                'prediction/grpc/test_protocol',
                'grpc test protocol prediction',
                self.server.model,
                self.server.test_loader,
                self.server.device,
                step=final_test_step,
                client_ids=list(self.expected_clients),
                state=self.server.global_state,
            )
        except Exception as exc:
            logger.debug('Skip gRPC prediction plot: {}', exc)
        self.tracker.finish()
        with (self.output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        logger.info('gRPC federated training finished: {}', summary)

    def submit_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Accept one remote client update and aggregate when all clients arrive."""

        with self.lock:
            if self.stopped:
                return {'accepted': False, 'stop': True, 'round': self.round_index}
            if not (self.runtime_ready and self.registration_ready):
                return {'accepted': False, 'stop': False, 'round': self.round_index}
            if payload['round'] != self.round_index:
                return {'accepted': False, 'stop': False, 'round': self.round_index}
            result = payload['result']
            self.pending[result.client_id] = result
            logger.info('Received update from {} for round {}', result.client_id, self.round_index)
            if set(self.expected_clients).issubset(self.pending.keys()):
                results = [self.pending[client_id] for client_id in self.expected_clients]
                round_base_state = _clone_state(self.server.global_state)
                round_context = self.server.build_round_context()
                if self.compressed:
                    aggregation_weights = self.server.aggregate_sparse(results, round_base_state=round_base_state, round_index=self.round_index, round_context=round_context)
                else:
                    aggregation_weights = self.server.aggregate_dense(results, round_index=self.round_index, round_base_state=round_base_state, round_context=round_context)
                next_round_index = self.round_index + 1
                if next_round_index >= self.max_rounds:
                    current_round_index = self.round_index
                    self.pending = {}
                    self.round_index = next_round_index
                    self.round_start_time = time.perf_counter()
                    self.stopped = True
                    self.stop_time_seconds = time.perf_counter()
                    self.finalize_requested = True
                    self.pending_final_round_thread = threading.Thread(
                        target=self._finish_final_round_bookkeeping,
                        args=(current_round_index, results, aggregation_weights, round_base_state, round_context),
                        name='grpc-final-round',
                        daemon=True,
                    )
                    self.pending_final_round_thread.start()
                else:
                    metrics = self.server.evaluate_global()
                    protocol_metrics = metrics
                    self.best_global_state, self.best_metrics, self.best_round, improved = _update_best_checkpoint(
                        best_state=self.best_global_state,
                        best_metrics=self.best_metrics,
                        best_index=self.best_round,
                        candidate_state=self.server.global_state,
                        candidate_metrics=metrics,
                        candidate_index=self.round_index,
                        label='round',
                        metric_name=self.primary_metric_name,
                        metric_mode=self.primary_metric_mode,
                    )
                    will_stop = self.stopper.update(metrics[self.primary_metric_name])
                    record = self.server.record_round(
                        self.round_index,
                        results,
                        aggregation_weights,
                        metrics,
                        round_time_seconds=time.perf_counter() - self.round_start_time,
                        elapsed_time_seconds=time.perf_counter() - self.start_time,
                        protocol_metrics=protocol_metrics,
                                    silent=will_stop,
                    )
                    if not will_stop:
                        self.tracker.log({**_wandb_round_payload(record), **_wandb_cumulative_communication_payload(self.server.history)}, step=self.round_index)
                    captured_update_records = _capture_round_update_records(
                        self.config,
                        self.capture_client_ids,
                        results,
                        self.round_index,
                        self.max_rounds,
                        round_base_state,
                        server=self.server,
                        round_context=round_context,
                    )
                    save_captured_update_records(self.output_dir, captured_update_records)
                    if (not will_stop) or should_save_periodic_artifacts(self.config, self.round_index + 1):
                        _save_periodic_federated_snapshot(
                            output_dir=self.output_dir,
                            config=self.config,
                            server=self.server,
                            round_index=self.round_index,
                            start_time=self.start_time,
                            best_global_state=self.best_global_state,
                                        best_metrics=self.best_metrics,
                            best_round=self.best_round,
                            transport='grpc',
                        )
                    self.pending = {}
                    self.round_index = next_round_index
                    self.round_start_time = time.perf_counter()
                    self.stopped = will_stop
                    if self.stopped:
                        self.stop_time_seconds = time.perf_counter()
                        self.finalize_requested = True
            return {'accepted': True, 'stop': self.stopped, 'round': self.round_index}


def serve(config: dict[str, Any]) -> None:
    """Start a blocking gRPC federated server."""

    config = copy.deepcopy(config)
    grpc_cfg = config.setdefault('grpc', {})
    grpc_cfg['defer_server_runtime_init'] = True
    address = grpc_cfg.get('address', '0.0.0.0:50051')
    poll_seconds = float(grpc_cfg.get('poll_seconds', 1.0))
    shutdown_grace_seconds = float(grpc_cfg.get('shutdown_grace_seconds', max(3.0, poll_seconds * 3.0)))
    shutdown_ack_timeout_seconds = float(
        grpc_cfg.get('shutdown_ack_timeout_seconds', max(shutdown_grace_seconds, poll_seconds * 20.0))
    )
    max_message_mb = float(grpc_cfg.get('max_message_mb', 256.0))
    max_message_length = int(max_message_mb * 1024 * 1024)
    coordinator = GrpcFederatedCoordinator(config)
    rpc_server = FederatedRpcServer(
        address,
        coordinator.get_global,
        coordinator.submit_update,
        coordinator.ack_stop,
        coordinator.register_client,
        max_message_length=max_message_length,
    )
    rpc_server.start()
    coordinator.start_runtime_initialization()
    try:
        while True:
            coordinator.finalize_if_requested()
            if coordinator.stopped and coordinator.finalization_completed:
                break
            time.sleep(poll_seconds)
        logger.info('gRPC server entering shutdown grace window of {:.2f}s', shutdown_grace_seconds)
        time.sleep(shutdown_grace_seconds)
        shutdown_deadline = time.perf_counter() + shutdown_ack_timeout_seconds
        while not coordinator.ready_for_shutdown() and time.perf_counter() < shutdown_deadline:
            time.sleep(poll_seconds)
        if coordinator.ready_for_shutdown():
            logger.info('All clients acknowledged stop; shutting down gRPC server')
        else:
            logger.warning(
                'Timed out waiting for stop acknowledgements ({}/{})',
                len(coordinator.stop_acked_clients),
                len(coordinator.expected_clients),
            )
    finally:
        rpc_server.stop(0)



def run_client(config: dict[str, Any], client_id: str) -> None:
    """Run a gRPC federated client loop until the server reports stop."""

    address = config.get('grpc', {}).get('server_address', config.get('grpc', {}).get('address', '127.0.0.1:50051'))
    poll_seconds = float(config.get('grpc', {}).get('poll_seconds', 1.0))
    max_message_mb = float(config.get('grpc', {}).get('max_message_mb', 256.0))
    max_message_length = int(max_message_mb * 1024 * 1024)
    client_output_dir = Path(config['experiment']['output_dir']) / f'client_{client_id}'
    setup_logging(client_output_dir, config.get('runtime', {}).get('log_level', 'INFO'))
    config_formats = config.get('artifacts', {}).get('config_formats')
    saved_configs = save_experiment_config(config, client_output_dir, config_formats)
    logger.info('Client {} saved startup config artifacts: {}', client_id, [str(path) for path in saved_configs])
    configure_torch_runtime(config)
    device = resolve_device(config)
    configure_random_seed(config, device=device)
    validate_transport_modes(config)
    train_loader, local_train_samples = _build_grpc_client_training_state(config, client_id)
    client = FederatedClient(
        client_id,
        train_loader,
        config,
        device,
        total_train_samples=local_train_samples,
        total_clients=1,
        allow_ega_pretrain=False,
    )
    rpc = FederatedRpcClient(address, max_message_length=max_message_length)
    def _ack_stop() -> None:
        """Acknowledge the final stop signal back to the server."""

        try:
            rpc.ack_stop({'client_id': client_id})
        except Exception as exc:
            logger.warning('Client {} could not acknowledge stop to {}: {}', client_id, address, exc)

    while True:
        try:
            registration = rpc.register_client(client.registration_payload())
        except Exception as exc:
            logger.warning('Client {} could not register with {}: {}', client_id, address, exc)
            time.sleep(poll_seconds)
            continue
        if registration.get('accepted'):
            logger.info(
                'Client {} registered local_train_samples={} ({}/{}) registration_ready={}',
                client_id,
                local_train_samples,
                registration.get('registered_clients'),
                registration.get('expected_clients'),
                registration.get('registration_ready'),
            )
            break
        error = registration.get('error') or 'registration rejected'
        raise ValueError(f'Client {client_id} registration failed: {error}')

    last_submitted = -1
    last_waiting_ready_log = -1
    last_transport_snapshot = rpc.snapshot_counters()
    while True:
        try:
            global_payload = rpc.get_global()
        except Exception as exc:
            logger.warning('Client {} could not fetch global state from {}: {}', client_id, address, exc)
            time.sleep(poll_seconds)
            continue
        if global_payload['stop']:
            logger.info('Client {} received stop signal', client_id)
            _ack_stop()
            return
        if not global_payload.get('ready', True):
            runtime_error = global_payload.get('runtime_error')
            if runtime_error:
                logger.error('Client {} observed server runtime initialization failure: {}', client_id, runtime_error)
                time.sleep(poll_seconds)
                continue
            if last_waiting_ready_log != global_payload['round']:
                logger.info(
                    'Client {} waiting for server readiness before round {} runtime_ready={} registration_ready={} registered={}/{}',
                    client_id,
                    global_payload['round'],
                    global_payload.get('runtime_ready'),
                    global_payload.get('registration_ready'),
                    global_payload.get('registered_clients'),
                    global_payload.get('expected_clients'),
                )
                last_waiting_ready_log = global_payload['round']
            time.sleep(poll_seconds)
            continue
        round_index = global_payload['round']
        if round_index == last_submitted:
            time.sleep(poll_seconds)
            continue
        logger.info(
            'Client {} fetched round {} compressed={} from {}',
            client_id,
            round_index,
            global_payload['compressed'],
            address,
        )
        train_start = time.perf_counter()
        result = client.train(
            global_payload['state'],
            compressed=global_payload['compressed'],
            round_index=round_index,
            round_context=global_payload.get('round_context'),
        )
        train_time_seconds = time.perf_counter() - train_start
        current_transport_snapshot = rpc.snapshot_counters()
        _apply_transport_metrics(result, _transport_delta(current_transport_snapshot, last_transport_snapshot), round_index=round_index)
        logger.info(
            'Client {} round {} local_loss={:.6f} train_time={:.2f}s payload={} compressor={} parameter_upload={} ({})/{} params parameter_download={} ({})/{} params transport_upload={} ({}) transport_download={} ({}) clip_norm={} noise_multiplier={}',
            client_id,
            round_index,
            result.loss,
            train_time_seconds,
            result.aggregation_payload_kind,
            result.compressor,
            result.parameter_upload_bytes,
            _format_num_bytes(result.parameter_upload_bytes),
            result.parameter_upload_parameters,
            result.parameter_download_bytes,
            _format_num_bytes(result.parameter_download_bytes),
            result.parameter_download_parameters,
            result.transport_upload_bytes,
            _format_num_bytes(result.transport_upload_bytes),
            result.transport_download_bytes,
            _format_num_bytes(result.transport_download_bytes),
            result.privacy_clip_norm,
            result.privacy_noise_multiplier,
        )
        while True:
            try:
                response = rpc.submit_update({'round': round_index, 'result': result})
                break
            except Exception as exc:
                logger.warning('Client {} could not submit round {} to {}: {}', client_id, round_index, address, exc)
                time.sleep(poll_seconds)
        logger.info(
            'Client {} round {} submit accepted={} stop={} server_round={}',
            client_id,
            round_index,
            response.get('accepted'),
            response.get('stop'),
            response.get('round'),
        )
        last_transport_snapshot = rpc.snapshot_counters()
        last_submitted = round_index if response.get('accepted') else last_submitted
        if response.get('stop'):
            logger.info('Client {} completed final round {}', client_id, round_index)
            _ack_stop()
            return
