"""Multi-process federated training built on the generic gRPC transport.

Example:
    Server:
        ``python scripts/server.py --config configs/test.yaml``

    Client:
        ``python scripts/client.py --client-id Nd2O3 --config configs/test.yaml``
"""

from __future__ import annotations

import json
import pickle
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

from federated_ts.communication.grpc_service import FederatedRpcClient, FederatedRpcServer
from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.federated.algorithms import (
    AsyncAttackManager,
    _attack_target_type,
    _build_attack_round_task,
    _clone_state,
    _round_history_communication_summary,
    _wandb_cumulative_communication_payload,
    _wandb_round_payload,
    configure_random_seed,
    configure_torch_runtime,
    resolve_device,
)
from federated_ts.federated.client import ClientResult, FederatedClient
from federated_ts.federated.server import EarlyStopper, FederatedServer
from federated_ts.security.attacks import summarize_attack_results
from federated_ts.utils.logging import setup_logging
from federated_ts.utils.serialization import state_num_bytes, state_num_parameters
from federated_ts.utils.tracking import Tracker


def _transport_delta(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    """Return the serialized transport delta between two RPC counter snapshots."""

    return {key: int(current.get(key, 0)) - int(previous.get(key, 0)) for key in current.keys()}



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
    estimated_upload_bytes = base_upload_bytes
    if round_index is not None:
        payload = {'round': round_index, 'result': result}
        for _ in range(3):
            payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            estimated_upload_bytes = base_upload_bytes + len(payload_bytes)
            if estimated_upload_bytes == result.transport_upload_bytes:
                break
            result.transport_upload_bytes = estimated_upload_bytes
            result.transport_upload_overhead_bytes = max(0, result.transport_upload_bytes - result.parameter_upload_bytes)
    else:
        result.transport_upload_bytes = base_upload_bytes
        result.transport_upload_overhead_bytes = max(0, result.transport_upload_bytes - result.parameter_upload_bytes)
    if round_index is not None:
        result.transport_upload_bytes = estimated_upload_bytes
        result.transport_upload_overhead_bytes = max(0, result.transport_upload_bytes - result.parameter_upload_bytes)
    return result


class GrpcFederatedCoordinator:
    """Server-side coordinator that aggregates updates when all clients report."""

    def __init__(self, config: dict[str, Any]):
        """Initialize server-side state for multi-process training."""

        self.config = config
        self.output_dir = Path(config['experiment']['output_dir'])
        setup_logging(self.output_dir, config.get('runtime', {}).get('log_level', 'INFO'))
        configure_torch_runtime(config)
        configure_random_seed(config)
        self.tracker = Tracker(config)
        self.device = resolve_device(config)
        train_loaders, val_loader, test_loader = build_federated_loaders(config)
        self.expected_clients = tuple(train_loaders.keys())
        self.attack_clients = [
            FederatedClient(client_id, loader, config, self.device)
            for client_id, loader in train_loaders.items()
        ]
        self.server = FederatedServer(config, val_loader, test_loader, self.device)
        self.compressed = str(config['federated'].get('algorithm', 'fedavg')).lower() in {
            'compressed_fedavg',
            'sparse_fedavg',
            'dp_topk_fedavg',
            'soteriafl',
        }
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
        self.lock = threading.Lock()
        self.attack_manager = AsyncAttackManager(config, self.tracker)
        self.attack_results = self.attack_manager.attack_results
        self.attack_target_type = _attack_target_type(config)
        self.tracker.log({
            'run/algorithm': str(config['federated'].get('algorithm', 'fedavg')),
            'run/client_count': len(self.expected_clients),
            'run/model_parameters': state_num_parameters(self.server.global_state),
            'run/model_bytes': state_num_bytes(self.server.global_state),
            'run/compressed_uploads': self.compressed,
            'run/transport': 'grpc',
        })

    def get_global(self) -> dict[str, Any]:
        """Return the current global payload for remote clients."""

        with self.lock:
            return {
                'round': self.round_index,
                'state': self.server.global_state,
                'compressed': self.compressed,
                'stop': self.stopped,
            }

    def _run_attacks(self, round_index: int, round_base_state: dict[str, Any], results) -> None:
        """Queue server-side attack evaluation without blocking aggregation or validation."""

        task = _build_attack_round_task(
            self.config,
            self.attack_clients,
            results,
            round_index,
            self.max_rounds,
            round_base_state,
            self.attack_target_type,
        )
        self.attack_manager.submit(task)

    def _finalize(self) -> None:
        """Persist final model artifacts and a summary compatible with the main FL path."""

        test_metrics = self.server.test_global()
        self.server.save(self.output_dir, self.config)
        self.attack_manager.finalize()
        total_elapsed = time.perf_counter() - self.start_time
        attack_records = [result.to_record() for result in self.attack_results]
        with (self.output_dir / 'attack_results.json').open('w', encoding='utf-8') as handle:
            json.dump(attack_records, handle, ensure_ascii=False, indent=2)
        attack_summary = summarize_attack_results(
            self.attack_results,
            float(self.config.get('attack', {}).get('success_rate_threshold', 0.03)),
        )
        summary = {
            'test': test_metrics,
            'rounds': len(self.server.history),
            'total_time_seconds': total_elapsed,
            'last_upload_compression_ratio': self.server.history[-1].upload_compression_ratio if self.server.history else 0.0,
            'last_total_communication_ratio': self.server.history[-1].total_communication_ratio if self.server.history else 0.0,
            'last_communication_ratio': self.server.history[-1].communication_ratio if self.server.history else 0.0,
            **_round_history_communication_summary(self.server.history),
            'transport': 'grpc',
            'attack_target_type': attack_summary.get('target_type', self.attack_target_type),
            'attack_primary_metric': attack_summary['primary_metric'],
            'attack_primary_metric_direction': attack_summary['primary_metric_direction'],
            'attack_overall_avg_mse': attack_summary['overall_avg_mse'],
            'attack_success_rate': attack_summary['overall_success_rate'],
            'attack_evaluations': len(attack_records),
            'attack_summary': attack_summary,
        }
        self.tracker.log({
            **{f'test/{key}': value for key, value in test_metrics.items()},
            'run/rounds': len(self.server.history),
            'run/total_time_seconds': total_elapsed,
            'run/transport': 'grpc',
        })
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
                if self.compressed:
                    aggregation_weights = self.server.aggregate_sparse(results)
                else:
                    aggregation_weights = self.server.aggregate_dense(results)
                metrics = self.server.evaluate_global()
                record = self.server.record_round(
                    self.round_index,
                    results,
                    aggregation_weights,
                    metrics,
                    round_time_seconds=time.perf_counter() - self.round_start_time,
                    elapsed_time_seconds=time.perf_counter() - self.start_time,
                )
                self.tracker.log({**_wandb_round_payload(record), **_wandb_cumulative_communication_payload(self.server.history)}, step=self.round_index)
                self._run_attacks(self.round_index, round_base_state, results)
                self.pending = {}
                self.round_index += 1
                self.round_start_time = time.perf_counter()
                self.stopped = self.round_index >= self.max_rounds or self.stopper.update(metrics['mse'])
                if self.stopped:
                    self._finalize()
            return {'accepted': True, 'stop': self.stopped, 'round': self.round_index}


def serve(config: dict[str, Any]) -> None:
    """Start a blocking gRPC federated server."""

    grpc_cfg = config.get('grpc', {})
    address = grpc_cfg.get('address', '0.0.0.0:50051')
    poll_seconds = float(grpc_cfg.get('poll_seconds', 1.0))
    shutdown_grace_seconds = float(grpc_cfg.get('shutdown_grace_seconds', max(3.0, poll_seconds * 3.0)))
    max_message_mb = float(grpc_cfg.get('max_message_mb', 256.0))
    max_message_length = int(max_message_mb * 1024 * 1024)
    coordinator = GrpcFederatedCoordinator(config)
    rpc_server = FederatedRpcServer(address, coordinator.get_global, coordinator.submit_update, max_message_length=max_message_length)
    rpc_server.start()
    try:
        while not coordinator.stopped:
            time.sleep(poll_seconds)
        logger.info('gRPC server entering shutdown grace window of {:.2f}s', shutdown_grace_seconds)
        time.sleep(shutdown_grace_seconds)
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
    device = resolve_device(config)
    train_loaders, _, _ = build_federated_loaders(config)
    if client_id not in train_loaders:
        raise ValueError(f'Unknown client_id {client_id}; expected one of {sorted(train_loaders)}')
    client = FederatedClient(client_id, train_loaders[client_id], config, device)
    rpc = FederatedRpcClient(address, max_message_length=max_message_length)
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
            return
        round_index = global_payload['round']
        if round_index == last_submitted:
            time.sleep(poll_seconds)
            continue
        result = client.train(global_payload['state'], compressed=global_payload['compressed'], round_index=round_index)
        current_transport_snapshot = rpc.snapshot_counters()
        _apply_transport_metrics(result, _transport_delta(current_transport_snapshot, last_transport_snapshot), round_index=round_index)
        while True:
            try:
                response = rpc.submit_update({'round': round_index, 'result': result})
                break
            except Exception as exc:
                logger.warning('Client {} could not submit round {} to {}: {}', client_id, round_index, address, exc)
                time.sleep(poll_seconds)
        last_transport_snapshot = rpc.snapshot_counters()
        last_submitted = round_index if response.get('accepted') else last_submitted
        if response.get('stop'):
            logger.info('Client {} completed final round {}', client_id, round_index)
            return
