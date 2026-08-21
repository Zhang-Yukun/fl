"""Multi-process federated training built on the generic gRPC transport.

Example:
    Server:
        ``python -m fedlab.entrypoints.server --config configs/test.yaml``

    Client:
        ``python -m fedlab.entrypoints.client --client-id Nd2O3 --config configs/test.yaml``
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from fedlab.communication.grpc_service import FederatedRpcClient, FederatedRpcServer
from fedlab.datasets import build_federated_loaders
from fedlab.federated.algorithms import (
    AsyncAttackManager,
    _attack_target_type,
    _build_federated_resume_state,
    _build_federated_summary,
    _resolve_test_metric_views,
    _build_attack_round_task,
    _clone_state,
    _round_history_communication_summary,
    _save_periodic_federated_snapshot,
    _update_best_checkpoint,
    _wandb_cumulative_communication_payload,
    _wandb_round_payload,
    configure_random_seed,
    configure_torch_runtime,
    resolve_device,
    is_compressed_algorithm,
)
from fedlab.federated.client import ClientResult, FederatedClient
from fedlab.federated.server import EarlyStopper, FederatedServer
from fedlab.federated.protocol import validate_transport_modes
from fedlab.security.attacks import save_attack_artifacts, summarize_attack_results
from fedlab.utils.logging import setup_logging
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
        configure_random_seed(config)
        validate_transport_modes(config, transport_backend='grpc')
        self.tracker = Tracker(config)
        self.device = resolve_device(config)
        train_loaders, val_loader, test_loader = build_federated_loaders(config)
        self.expected_clients = tuple(train_loaders.keys())
        self.server = FederatedServer(config, val_loader, test_loader, self.device)
        total_train_samples = sum(_loader_num_samples(loader) for loader in train_loaders.values())
        self.attack_clients = [
            FederatedClient(
                client_id,
                loader,
                config,
                self.device,
                total_train_samples=total_train_samples,
                total_clients=len(train_loaders),
                allow_ega_pretrain=False,
            )
            for client_id, loader in train_loaders.items()
        ]
        self.compressed = is_compressed_algorithm(config)
        self.max_rounds = int(config['federated'].get('rounds', 20))
        self.stopper = EarlyStopper(
            int(config['training'].get('patience', 5)),
            float(config['training'].get('min_delta', 0.0)),
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
        self.attack_manager = AsyncAttackManager(config, self.tracker)
        self.attack_results = self.attack_manager.attack_results
        self.attack_target_type = _attack_target_type(config)
        self.best_global_state = _clone_state(self.server.global_state)
        self.best_oracle_state = None if not self.server._uses_oracle_evaluation() else _clone_state(self.server.oracle_global_state)
        self.best_metrics = {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
        self.best_round = -1
        self.tracker.log({
            'run/algorithm': str(config['federated'].get('algorithm', 'fedavg')),
            'run/client_count': len(self.expected_clients),
            'run/model_parameters': state_num_parameters(self.server.global_state),
            'run/model_bytes': state_num_bytes(self.server.global_state),
            'run/compressed_uploads': self.compressed,
            'run/transport': 'grpc',
        })

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
            return {
                'round': self.round_index,
                'state': self.server.global_state,
                'compressed': self.compressed,
                'round_context': self.server.build_round_context(),
                'stop': self.stopped,
            }

    def _run_attacks(self, round_index: int, round_base_state: dict[str, Any], results, round_context: dict[str, Any] | None = None) -> None:
        """Queue server-side attack evaluation without blocking aggregation or validation."""

        task = _build_attack_round_task(
            self.config,
            self.attack_clients,
            results,
            round_index,
            self.max_rounds,
            round_base_state,
            self.attack_target_type,
            server=self.server,
            round_context=round_context,
        )
        self.attack_manager.submit(task)

    def _finish_final_round_bookkeeping(self, round_index: int, results, aggregation_weights, round_base_state, round_context: dict[str, Any] | None = None) -> None:
        """Complete final-round validation and artifact bookkeeping off the submit hot path."""

        metrics = self.server.evaluate_global()
        protocol_metrics = self.server.evaluate_protocol() if self.server._uses_oracle_evaluation() else metrics
        oracle_metrics = self.server.evaluate_oracle() if self.server._uses_oracle_evaluation() else protocol_metrics
        self.best_global_state, self.best_metrics, self.best_round, improved = _update_best_checkpoint(
            best_state=self.best_global_state,
            best_metrics=self.best_metrics,
            best_index=self.best_round,
            candidate_state=self.server.global_state,
            candidate_metrics=metrics,
            candidate_index=round_index,
            label='round',
        )
        if improved and self.server._uses_oracle_evaluation():
            self.best_oracle_state = _clone_state(self.server.oracle_global_state)
        record = self.server.record_round(
            round_index,
            results,
            aggregation_weights,
            metrics,
            round_time_seconds=time.perf_counter() - self.round_start_time,
            elapsed_time_seconds=time.perf_counter() - self.start_time,
            protocol_metrics=protocol_metrics,
            oracle_metrics=oracle_metrics,
            silent=True,
        )
        self.tracker.log({**_wandb_round_payload(record), **_wandb_cumulative_communication_payload(self.server.history)}, step=round_index)
        try:
            input_series, prediction, target = predict_first_batch_for_state(self.server.model, self.server.global_state, self.server.val_loader, self.server.device)
            self.tracker.log_prediction_plot('prediction/grpc/val_protocol', input_series, prediction, target, step=round_index, title='grpc val protocol prediction')
            oracle_state = self.server.oracle_global_state if self.server.oracle_global_state is not None else self.server.global_state
            input_series, prediction, target = predict_first_batch_for_state(self.server.model, oracle_state, self.server.val_loader, self.server.device)
            self.tracker.log_prediction_plot('prediction/grpc/val_oracle', input_series, prediction, target, step=round_index, title='grpc val oracle prediction')
        except Exception as exc:
            logger.debug('Skip gRPC val prediction plot: {}', exc)
        self._run_attacks(round_index, round_base_state, results, round_context)
        if should_save_periodic_artifacts(self.config, round_index + 1):
            _save_periodic_federated_snapshot(
                output_dir=self.output_dir,
                config=self.config,
                server=self.server,
                round_index=round_index,
                start_time=self.start_time,
                best_global_state=self.best_global_state,
                best_oracle_state=self.best_oracle_state,
                best_metrics=self.best_metrics,
                best_round=self.best_round,
                attack_results=self.attack_results,
                attack_target_type=self.attack_target_type,
                transport='grpc',
            )

    def _finalize(self) -> None:
        """Persist final model artifacts and a summary compatible with the main FL path."""

        self.server.global_state = _clone_state(self.best_global_state)
        if self.server._uses_oracle_evaluation() and self.best_oracle_state is not None:
            self.server.oracle_global_state = _clone_state(self.best_oracle_state)
        logger.info('Restored best gRPC federated checkpoint from round {} for final test', self.best_round)
        test_metrics = self.server.test_global()
        protocol_test_metrics, oracle_test_metrics = _resolve_test_metric_views(self.server, test_metrics)
        self.server.save(self.output_dir, self.config)
        final_test_step = max(len(self.server.history), self.best_round + 1)
        self.attack_manager.finalize()
        total_elapsed = time.perf_counter() - self.start_time
        attack_records = save_attack_artifacts(self.output_dir, self.attack_results)
        with (self.output_dir / 'attack_results.json').open('w', encoding='utf-8') as handle:
            json.dump(attack_records, handle, ensure_ascii=False, indent=2)
        attack_summary = summarize_attack_results(
            self.attack_results,
            float(self.config.get('attack', {}).get('success_rate_threshold', 0.03)),
        )
        summary = _build_federated_summary(
            server=self.server,
            test_metrics=test_metrics,
            total_elapsed=total_elapsed,
            best_round=self.best_round,
            best_metrics=self.best_metrics,
            attack_records=attack_records,
            attack_summary=attack_summary,
            attack_target_type=self.attack_target_type,
            protocol_test_metrics=protocol_test_metrics,
            oracle_test_metrics=oracle_test_metrics,
            transport='grpc',
        )
        final_log_payload = {
            **{f'test/{key}': value for key, value in test_metrics.items()},
            'run/rounds': len(self.server.history),
            'run/total_time_seconds': total_elapsed,
            'run/transport': 'grpc',
            'run/evaluation_mode': self.server.evaluation_mode,
            'run/best_round': self.best_round,
            'run/best_val_mse': self.best_metrics['mse'],
            'run/best_val_mae': self.best_metrics['mae'],
            'run/best_val_mape': self.best_metrics['mape'],
            'privacy/epsilon': summary['privacy_epsilon'],
            'privacy/delta': summary['privacy_delta'],
            'privacy/rdp_total': summary['privacy_rdp_total'],
            'privacy/sampling_rate': summary['privacy_sampling_rate'],
            'privacy/adaptive_clip_norm': summary['adaptive_clip_norm'],
        }
        if protocol_test_metrics is not None:
            final_log_payload.update({f'protocol_test/{key}': value for key, value in protocol_test_metrics.items()})
        if oracle_test_metrics is not None:
            final_log_payload.update({f'oracle_test/{key}': value for key, value in oracle_test_metrics.items()})
        self.tracker.log(final_log_payload)
        try:
            input_series, prediction, target = predict_first_batch_for_state(self.server.model, self.server.global_state, self.server.test_loader, self.server.device)
            self.tracker.log_prediction_plot('prediction/grpc/test_protocol', input_series, prediction, target, step=final_test_step, title='grpc test protocol prediction')
            oracle_state = self.server.oracle_global_state if self.server.oracle_global_state is not None else self.server.global_state
            input_series, prediction, target = predict_first_batch_for_state(self.server.model, oracle_state, self.server.test_loader, self.server.device)
            self.tracker.log_prediction_plot('prediction/grpc/test_oracle', input_series, prediction, target, step=final_test_step, title='grpc test oracle prediction')
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
                    protocol_metrics = self.server.evaluate_protocol() if self.server._uses_oracle_evaluation() else metrics
                    oracle_metrics = self.server.evaluate_oracle() if self.server._uses_oracle_evaluation() else None
                    self.best_global_state, self.best_metrics, self.best_round, improved = _update_best_checkpoint(
                        best_state=self.best_global_state,
                        best_metrics=self.best_metrics,
                        best_index=self.best_round,
                        candidate_state=self.server.global_state,
                        candidate_metrics=metrics,
                        candidate_index=self.round_index,
                        label='round',
                    )
                    if improved and self.server._uses_oracle_evaluation():
                        self.best_oracle_state = _clone_state(self.server.oracle_global_state)
                    will_stop = self.stopper.update(metrics['mse'])
                    record = self.server.record_round(
                        self.round_index,
                        results,
                        aggregation_weights,
                        metrics,
                        round_time_seconds=time.perf_counter() - self.round_start_time,
                        elapsed_time_seconds=time.perf_counter() - self.start_time,
                        protocol_metrics=protocol_metrics,
                        oracle_metrics=oracle_metrics,
                        silent=will_stop,
                    )
                    if not will_stop:
                        self.tracker.log({**_wandb_round_payload(record), **_wandb_cumulative_communication_payload(self.server.history)}, step=self.round_index)
                    self._run_attacks(self.round_index, round_base_state, results, round_context)
                    if (not will_stop) or should_save_periodic_artifacts(self.config, self.round_index + 1):
                        _save_periodic_federated_snapshot(
                            output_dir=self.output_dir,
                            config=self.config,
                            server=self.server,
                            round_index=self.round_index,
                            start_time=self.start_time,
                            best_global_state=self.best_global_state,
                            best_oracle_state=self.best_oracle_state,
                            best_metrics=self.best_metrics,
                            best_round=self.best_round,
                            attack_results=self.attack_results,
                            attack_target_type=self.attack_target_type,
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

    grpc_cfg = config.get('grpc', {})
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
        max_message_length=max_message_length,
    )
    rpc_server.start()
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
    setup_logging(Path(config['experiment']['output_dir']) / f'client_{client_id}', config.get('runtime', {}).get('log_level', 'INFO'))
    configure_torch_runtime(config)
    configure_random_seed(config)
    validate_transport_modes(config, transport_backend='grpc')
    device = resolve_device(config)
    train_loaders, _, _ = build_federated_loaders(config)
    if client_id not in train_loaders:
        raise ValueError(f'Unknown client_id {client_id}; expected one of {sorted(train_loaders)}')
    total_train_samples = sum(_loader_num_samples(loader) for loader in train_loaders.values())
    client = FederatedClient(
        client_id,
        train_loaders[client_id],
        config,
        device,
        total_train_samples=total_train_samples,
        total_clients=len(train_loaders),
        allow_ega_pretrain=False,
    )
    rpc = FederatedRpcClient(address, max_message_length=max_message_length)
    def _ack_stop() -> None:
        """Acknowledge the final stop signal back to the server."""

        try:
            rpc.ack_stop({'client_id': client_id})
        except Exception as exc:
            logger.warning('Client {} could not acknowledge stop to {}: {}', client_id, address, exc)

    last_submitted = -1
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
