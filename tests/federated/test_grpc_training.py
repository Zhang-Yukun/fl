import json
import multiprocessing
import socket
import time
from threading import Event
from pathlib import Path

import pytest
import torch

import fedlab.federated.server as server_module
import fedlab.federated.client as client_module
import fedlab.communication.grpc_training as grpc_training_module
from fedlab.communication.grpc_training import GrpcFederatedCoordinator, _apply_transport_metrics, run_client, serve
from fedlab.datasets import build_federated_loaders
import fedlab.federated.algorithms as algorithms_module
from fedlab.replay_capture.artifacts import load_captured_update_records
import fedlab.federated.methods.encoded as encoded_methods
from fedlab.federated.algorithms import resolve_device, run_federated
from fedlab.federated.client import ClientResult, FederatedClient
from fedlab.utils.config import load_config
from fedlab.utils.serialization import compress_topk, serialize_model
from fedlab.utils.consistency import compare_fedavg_runs


def _register_loaded_clients(coordinator: GrpcFederatedCoordinator, train_loaders: dict[str, object]) -> None:
    """Register in-process test clients with their local sample counts."""

    for client_id, loader in train_loaders.items():
        response = coordinator.register_client({'client_id': client_id, 'local_train_samples': len(loader.dataset)})
        assert response['accepted'] is True


def _submit_one_round(coordinator: GrpcFederatedCoordinator, config: dict[str, object]) -> dict[str, object]:
    """Submit one full gRPC round using in-process clients for tests."""

    train_loaders, _, _ = build_federated_loaders(config)
    _register_loaded_clients(coordinator, train_loaders)
    device = resolve_device(config)
    global_payload = coordinator.get_global()
    response = None
    for client_id, loader in train_loaders.items():
        client = FederatedClient(client_id, loader, config, device)
        result = client.train(
            global_payload["state"],
            compressed=global_payload["compressed"],
            round_index=global_payload["round"],
            round_context=global_payload.get("round_context", {}),
        )
        response = coordinator.submit_update({"round": global_payload["round"], "result": result})
    assert response is not None
    while coordinator.finalize_requested and not coordinator.finalization_completed:
        coordinator.finalize_if_requested()
        if not coordinator.finalization_completed:
            time.sleep(0.01)
    return response


def _free_port() -> int:
    """Return one temporary localhost TCP port for gRPC integration tests."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_networked_grpc(config: dict[str, object]) -> None:
    """Run the real gRPC transport with local multi-process server and clients."""

    ctx = multiprocessing.get_context("spawn")
    server_process = ctx.Process(target=serve, args=(config,), name="grpc-server")
    client_processes = [
        ctx.Process(target=run_client, args=(config, client_id), name=f"grpc-client-{client_id}")
        for client_id in config["data"]["clients"]
    ]
    server_process.start()
    try:
        time.sleep(1.0)
        for process in client_processes:
            process.start()
        for process in client_processes:
            process.join(timeout=180)
            assert process.exitcode == 0, f"client process failed: {process.name} exitcode={process.exitcode}"
        server_process.join(timeout=180)
        assert server_process.exitcode == 0, f"server process failed: exitcode={server_process.exitcode}"
    finally:
        for process in client_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if server_process.is_alive():
            server_process.terminate()
            server_process.join(timeout=5)


def _write_classification_split(root: Path, client_id: str, split: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    client_dir = root / "clients" / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, client_dir / f"{split}.pt")


def _prepare_classification_split_dir(root: Path, *, client_ids: list[str], image_shape: tuple[int, int, int], num_classes: int) -> None:
    train_images_all = []
    train_labels_all = []
    val_images_all = []
    val_labels_all = []
    test_images_all = []
    test_labels_all = []
    for client_offset, client_id in enumerate(client_ids):
        base_value = float(client_offset) / 10.0
        train_images = torch.full((6, *image_shape), base_value, dtype=torch.float32)
        train_labels = torch.tensor([index % num_classes for index in range(6)], dtype=torch.long)
        val_images = torch.full((3, *image_shape), base_value + 0.1, dtype=torch.float32)
        val_labels = torch.tensor([index % num_classes for index in range(3)], dtype=torch.long)
        test_images = torch.full((3, *image_shape), base_value + 0.2, dtype=torch.float32)
        test_labels = torch.tensor([index % num_classes for index in range(3)], dtype=torch.long)
        _write_classification_split(root, client_id, 'train', train_images, train_labels)
        _write_classification_split(root, client_id, 'val', val_images, val_labels)
        _write_classification_split(root, client_id, 'test', test_images, test_labels)
        train_images_all.append(train_images)
        train_labels_all.append(train_labels)
        val_images_all.append(val_images)
        val_labels_all.append(val_labels)
        test_images_all.append(test_images)
        test_labels_all.append(test_labels)
    server_dir = root / 'server'
    server_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': torch.cat(train_images_all, dim=0), 'labels': torch.cat(train_labels_all, dim=0)}, server_dir / 'train.pt')
    torch.save({'images': torch.cat(val_images_all, dim=0), 'labels': torch.cat(val_labels_all, dim=0)}, server_dir / 'val.pt')
    torch.save({'images': torch.cat(test_images_all, dim=0), 'labels': torch.cat(test_labels_all, dim=0)}, server_dir / 'test.pt')


class _IdentityEgaCodec(torch.nn.Module):
    """Minimal codec that avoids real EGA artifact pretraining in gRPC tests."""

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = int(block_size)
        self.encoded_dim = int(block_size)
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        return blocks.to(torch.float32)

    def decode_blocks(self, encoded_blocks: torch.Tensor) -> torch.Tensor:
        return encoded_blocks.to(torch.float32)


def _patch_identity_ega_codec(monkeypatch, *, block_size: int = 8) -> None:
    def _fake_load_ega_codec(config, device, num_clients, allow_pretrain):
        del config, device, num_clients, allow_pretrain
        return _IdentityEgaCodec(block_size=block_size)

    def _fake_load_ega_codec_payload(config, payload, device, num_clients):
        del config, payload, device, num_clients
        return _IdentityEgaCodec(block_size=block_size)

    monkeypatch.setattr(encoded_methods, 'load_ega_codec', _fake_load_ega_codec)
    monkeypatch.setattr(encoded_methods, 'load_ega_codec_payload', _fake_load_ega_codec_payload)


def _classification_grpc_config(tmp_path: Path, *, algorithm: str) -> dict[str, object]:
    split_dir = tmp_path / 'classification_split'
    _prepare_classification_split_dir(split_dir, client_ids=['m1', 'm2', 'm3'], image_shape=(1, 4, 4), num_classes=3)
    config = {
        'experiment': {'output_dir': str(tmp_path / algorithm), 'mode': 'federated'},
        'runtime': {'device': 'cpu', 'log_level': 'INFO', 'deterministic': True, 'seed': 2026},
        'task': {'type': 'classification'},
        'data': {
            'split_dir': str(split_dir),
            'clients': ['m1', 'm2', 'm3'],
            'batch_size': 2,
            'shuffle_train': False,
            'num_workers': 0,
            'image_shape': [1, 4, 4],
            'num_classes': 3,
        },
        'model': {'name': 'small_cnn', 'hidden_channels': 4, 'dropout': 0.0},
        'training': {'epochs': 1, 'lr': 0.001, 'optimizer': 'adam', 'loss': 'cross_entropy', 'patience': 1, 'min_delta': 0.0},
        'centralized': {},
        'federated': {'algorithm': algorithm, 'rounds': 1},
        'replay_capture': {'enabled': True, 'frequency_rounds': 1},
        'attack': {'enabled': False, 'async_enabled': False, 'device': 'cpu'},
        'tracking': {'enabled': False},
        'evaluation': {'metrics': ['accuracy']},
        'artifacts': {'config_formats': ['yaml'], 'save_every_rounds': 0},
        'grpc': {'address': '127.0.0.1:50051', 'server_address': '127.0.0.1:50051', 'poll_seconds': 0.05, 'max_message_mb': 32.0},
    }
    if algorithm == 'sparse_fedavg':
        config['federated']['topk_fraction'] = 1.0
    if algorithm == 'ega_fedavg':
        config['federated']['quantization_seed'] = 2026
        config['ega'] = {
            'artifact_path': str(tmp_path / 'artifacts' / 'ega_codec.pt'),
            'block_size': 8,
            'encoded_dim': 8,
            'hidden_dim': 8,
            'residual_blocks': 0,
            'quantization_level': 64,
            'encoded_dtype': 'float32',
            'error_feedback': False,
            'min_normalization': 1e-6,
        }
    return config


def test_apply_transport_metrics_tracks_overhead_bytes():
    """Transport counters should be recorded separately from parameter payload bytes."""

    result = ClientResult(
        client_id="Nd2O3",
        num_samples=4,
        loss=0.1,
        parameter_download_bytes=100,
        parameter_download_parameters=25,
        parameter_upload_bytes=120,
        parameter_upload_parameters=30,
        transport_upload_bytes=0,
    )

    updated = _apply_transport_metrics(result, {"sent_bytes": 16, "received_bytes": 140}, round_index=0)

    assert updated.transport_download_bytes == 140
    assert updated.transport_download_overhead_bytes == 40
    assert updated.transport_upload_bytes >= updated.parameter_upload_bytes
    assert updated.transport_upload_overhead_bytes >= 0


def test_grpc_coordinator_saves_config_on_startup(tmp_path):
    """The gRPC coordinator should persist config artifacts before rounds begin."""

    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )

    coordinator = GrpcFederatedCoordinator(config)

    assert coordinator.output_dir == tmp_path
    assert (tmp_path / "config.yaml").exists()


def test_grpc_coordinator_waits_for_stop_acks(tmp_path):
    """The gRPC coordinator should wait for explicit client stop acknowledgements."""

    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=fedavg",
            "federated.rounds=1",
                    "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    coordinator = GrpcFederatedCoordinator(config)
    response = _submit_one_round(coordinator, config)

    assert response["stop"] is True
    assert coordinator.ready_for_shutdown() is False

    ack = coordinator.ack_stop({"client_id": "Nd2O3"})
    assert ack["acked_clients"] == 1
    assert coordinator.ready_for_shutdown() is False

    coordinator.ack_stop({"client_id": "CeO2"})
    coordinator.ack_stop({"client_id": "La2O3"})
    assert coordinator.ready_for_shutdown() is True


def test_grpc_coordinator_aggregates_one_round_and_saves_summary(tmp_path):
    """The gRPC coordinator accepts one round of client updates and persists summary artifacts."""

    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=fedavg",
            "federated.rounds=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    coordinator = GrpcFederatedCoordinator(config)
    response = _submit_one_round(coordinator, config)

    assert response["accepted"] is True
    assert response["stop"] is True
    assert response["round"] == 1
    assert coordinator.stopped is True
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "summary.json").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert len(metrics) == 1
    assert metrics[0]["algorithm"] == "fedavg"
    assert metrics[0]["total_parameter_upload_bytes"] == metrics[0]["total_upload_bytes"]
    assert metrics[0]["total_transport_upload_bytes"] >= metrics[0]["total_parameter_upload_bytes"]
    assert summary["transport"] == "grpc"
    assert summary["rounds"] == 1
    assert summary["last_parameter_download_compression_ratio"] < 1.0
    assert summary["last_parameter_upload_compression_ratio"] == 1.0
    assert metrics[0]["parameter_download_compression_ratio"] < 1.0
    assert metrics[0]["parameter_upload_compression_ratio"] == 1.0
    assert summary["total_parameter_bytes"] == metrics[0]["total_parameter_bytes"]
    assert summary["total_transport_bytes"] == metrics[0]["total_transport_bytes"]


def test_grpc_coordinator_saves_periodic_snapshot(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=fedavg",
            "federated.rounds=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "artifacts.save_every_rounds=1",
        ],
    )
    coordinator = GrpcFederatedCoordinator(config)
    response = _submit_one_round(coordinator, config)

    assert response["stop"] is True
    snapshot_dir = tmp_path / "snapshots" / "round_0001"
    assert (snapshot_dir / "config.yaml").exists()
    assert (snapshot_dir / "metrics.json").exists()
    assert (snapshot_dir / "summary.json").exists()
    assert (snapshot_dir / "model.pt").exists()
    assert (snapshot_dir / "resume_state.pt").exists()


def test_grpc_coordinator_does_not_persist_online_attack_results(tmp_path):
    """The gRPC coordinator should only persist update captures for offline replay."""

    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=fedavg",
            "federated.rounds=1",
            "attack.enabled=true",
            "attack.target_type=update_payload",
            "replay_capture.frequency_rounds=1",
            "attack.max_samples=1",
            "attack.clients_per_round=1",
            "attack.steps=1",
            "attack.optimizer=adam",
            "attack.local_optimizer=adam",
            "attack.async_enabled=true",
            "attack.async_workers=1",
            "attack.device=cpu",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "data.shuffle_train=false",
            "model.dropout=0.0",
        ],
    )
    coordinator = GrpcFederatedCoordinator(config)
    response = _submit_one_round(coordinator, config)

    assert response["stop"] is True
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert 'attack_evaluations' not in summary
    assert not (tmp_path / 'attack_results.json').exists()
    assert not (tmp_path / 'attack_artifacts').exists()


def test_grpc_coordinator_saves_update_captures_when_attacks_disabled(tmp_path):
    """The gRPC server should persist per-client update captures for offline replay."""

    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=fedavg",
            "federated.rounds=1",
            "attack.enabled=false",
            "replay_capture.frequency_rounds=1",
            "attack.max_samples=1",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "data.shuffle_train=false",
            "model.dropout=0.0",
        ],
    )
    coordinator = GrpcFederatedCoordinator(config)
    response = _submit_one_round(coordinator, config)

    assert response["stop"] is True
    captures = load_captured_update_records(tmp_path)
    assert len(captures) == 3
    assert {record["client_id"] for record in captures} == {"Nd2O3", "CeO2", "La2O3"}
    assert (tmp_path / "saved_updates" / "index.json").exists()
    assert "reference_inputs" not in captures[0]
    assert "reference_targets" not in captures[0]


def test_grpc_coordinator_defers_ega_runtime_until_explicit_start(tmp_path, monkeypatch):
    config = _classification_grpc_config(tmp_path, algorithm='ega_fedavg')
    config['grpc']['defer_server_runtime_init'] = True

    ready_event = Event()

    def _delayed_load_ega_codec(config, device, num_clients, allow_pretrain):
        del config, device, num_clients, allow_pretrain
        assert ready_event.wait(timeout=1.0)
        return _IdentityEgaCodec(block_size=8)

    monkeypatch.setattr(encoded_methods, 'load_ega_codec', _delayed_load_ega_codec)

    coordinator = GrpcFederatedCoordinator(config)
    train_loaders, _, _ = build_federated_loaders(config)
    _register_loaded_clients(coordinator, train_loaders)
    initial_payload = coordinator.get_global()
    assert initial_payload['ready'] is False
    assert initial_payload['state'] is None
    assert initial_payload['round_context'] == {}

    coordinator.start_runtime_initialization()
    time.sleep(0.05)
    waiting_payload = coordinator.get_global()
    assert waiting_payload['ready'] is False
    assert waiting_payload['state'] is None

    ready_event.set()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        payload = coordinator.get_global()
        if payload['ready']:
            assert 'ega_codec_payload' in payload['round_context']
            break
        time.sleep(0.01)
    else:  # pragma: no cover - defensive timeout path
        pytest.fail('deferred EGA runtime initialization did not complete in time')


def test_grpc_registration_records_uploaded_scaler_metadata(tmp_path):
    split_dir = tmp_path / 'rare_split'
    for client, start_offset in [('Nd2O3', 0), ('CeO2', 100), ('La2O3', 200)]:
        client_dir = split_dir / 'clients' / client
        client_dir.mkdir(parents=True, exist_ok=True)
        for split, start, length in [('train', start_offset, 40), ('val', start_offset + 40, 20), ('test', start_offset + 60, 20)]:
            dates = [f'2020-01-{(index % 28) + 1:02d}' for index in range(start, start + length)]
            values = torch.arange(start, start + length, dtype=torch.float32).tolist()
            rows = 'date,value\n' + '\n'.join(f'{date},{value}' for date, value in zip(dates, values)) + '\n'
            (client_dir / f'{split}.csv').write_text(rows, encoding='utf-8')

    config = {
        'experiment': {'output_dir': str(tmp_path / 'forecasting_server'), 'mode': 'federated'},
        'runtime': {'device': 'cpu', 'log_level': 'INFO', 'deterministic': True, 'seed': 2026},
        'task': {'type': 'forecasting'},
        'data': {
            'split_dir': str(split_dir),
            'clients': ['Nd2O3', 'CeO2', 'La2O3'],
            'seq_len': 4,
            'pred_len': 2,
            'batch_size': 8,
            'shuffle_train': False,
            'num_workers': 0,
        },
        'model': {'name': 'lstm', 'input_dim': 1, 'hidden_dim': 8, 'num_layers': 1, 'dropout': 0.0, 'pred_len': 2},
        'training': {'epochs': 1, 'lr': 0.001, 'optimizer': 'adam', 'loss': 'mse', 'patience': 1, 'min_delta': 0.0},
        'centralized': {},
        'federated': {'algorithm': 'fedavg', 'rounds': 1},
        'replay_capture': {'enabled': True, 'frequency_rounds': 1},
        'attack': {'enabled': False, 'async_enabled': False, 'device': 'cpu'},
        'tracking': {'enabled': False},
        'evaluation': {'metrics': ['mse', 'mae', 'mape']},
        'artifacts': {'config_formats': ['yaml'], 'save_every_rounds': 0},
        'grpc': {'address': '127.0.0.1:50051', 'server_address': '127.0.0.1:50051', 'poll_seconds': 0.05, 'max_message_mb': 32.0},
    }

    coordinator = GrpcFederatedCoordinator(config)
    train_loaders, _, _ = build_federated_loaders(config)
    payload = FederatedClient('Nd2O3', train_loaders['Nd2O3'], config, resolve_device(config)).registration_payload()

    response = coordinator.register_client(payload)

    assert response['accepted'] is True
    assert coordinator.registered_client_metadata['Nd2O3']['local_train_samples'] == 35
    assert coordinator.registered_client_metadata['Nd2O3']['scale_mean'] == pytest.approx([19.5])
    assert coordinator.registered_client_metadata['Nd2O3']['scale_std'] == pytest.approx([11.54339599609375])


def test_grpc_round_context_broadcasts_registered_global_training_stats(tmp_path):
    config = _classification_grpc_config(tmp_path, algorithm='fedavg')

    coordinator = GrpcFederatedCoordinator(config)
    train_loaders, _, _ = build_federated_loaders(config)
    _register_loaded_clients(coordinator, train_loaders)

    payload = coordinator.get_global()

    assert payload['ready'] is True
    assert payload['registration_ready'] is True
    assert payload['round_context']['total_clients'] == 3
    assert payload['round_context']['total_train_samples'] == 18



@pytest.mark.parametrize('algorithm', ['fedavg', 'sparse_fedavg', 'ega_fedavg'])
def test_grpc_coordinator_supports_classification_task(tmp_path, monkeypatch, algorithm):
    config = _classification_grpc_config(tmp_path, algorithm=algorithm)
    if algorithm == 'ega_fedavg':
        _patch_identity_ega_codec(monkeypatch)

    coordinator = GrpcFederatedCoordinator(config)
    response = _submit_one_round(coordinator, config)

    assert response['accepted'] is True
    assert response['stop'] is True
    summary = json.loads((Path(config['experiment']['output_dir']) / 'summary.json').read_text(encoding='utf-8'))
    metrics = json.loads((Path(config['experiment']['output_dir']) / 'metrics.json').read_text(encoding='utf-8'))
    captures = load_captured_update_records(Path(config['experiment']['output_dir']))

    assert summary['transport'] == 'grpc'
    assert 'accuracy' in summary['test']
    assert metrics[0]['algorithm'] == algorithm
    assert len(captures) == 3


def test_grpc_classification_uses_primary_metric_for_multi_round_stopping(tmp_path):
    config = _classification_grpc_config(tmp_path, algorithm='fedavg')
    config['federated']['rounds'] = 2

    coordinator = GrpcFederatedCoordinator(config)

    first_response = _submit_one_round(coordinator, config)
    assert first_response['accepted'] is True
    assert first_response['stop'] is False
    assert coordinator.round_index == 1

    second_response = _submit_one_round(coordinator, config)
    assert second_response['accepted'] is True
    assert second_response['stop'] is True

    summary = json.loads((Path(config['experiment']['output_dir']) / 'summary.json').read_text(encoding='utf-8'))
    metrics = json.loads((Path(config['experiment']['output_dir']) / 'metrics.json').read_text(encoding='utf-8'))

    assert summary['best_val_metric_name'] == 'accuracy'
    assert 'accuracy' in summary['best_val']
    assert len(metrics) == 2
    assert all('accuracy' in entry['val_metrics'] for entry in metrics)



def test_grpc_matches_single_process_fedavg_when_randomness_disabled(tmp_path):
    """Single-process FedAvg and the sequential gRPC coordinator should agree under deterministic settings."""

    local_dir = tmp_path / "local"
    grpc_dir = tmp_path / "grpc"
    overrides = [
        "federated.algorithm=fedavg",
        "federated.rounds=1",
        "training.patience=1",
        "attack.enabled=true",
        "attack.target_type=update_payload",
        "replay_capture.frequency_rounds=1",
        "attack.max_samples=1",
        "attack.clients_per_round=1",
        "attack.client_selection=first",
        "attack.steps=1",
        "attack.optimizer=adam",
        "attack.local_optimizer=adam",
        "attack.async_enabled=true",
        "attack.async_workers=1",
        "attack.device=cpu",
        "tracking.enabled=false",
        "runtime.device=cpu",
        "runtime.seed=2026",
        "runtime.deterministic=true",
        "data.shuffle_train=false",
        "model.dropout=0.0",
    ]
    local_config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        ["experiment.output_dir=" + str(local_dir), *overrides],
    )
    grpc_config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        ["experiment.output_dir=" + str(grpc_dir), *overrides],
    )

    local_summary = run_federated(local_config)
    coordinator = GrpcFederatedCoordinator(grpc_config)
    response = _submit_one_round(coordinator, grpc_config)
    assert response["stop"] is True

    local_metrics = json.loads((local_dir / "metrics.json").read_text(encoding="utf-8"))
    grpc_metrics = json.loads((grpc_dir / "metrics.json").read_text(encoding="utf-8"))
    grpc_summary = json.loads((grpc_dir / "summary.json").read_text(encoding="utf-8"))

    assert sorted((local_dir / "saved_updates").rglob("round_*.pt"))
    assert sorted((grpc_dir / "saved_updates").rglob("round_*.pt"))
    assert local_metrics[0]["algorithm"] == grpc_metrics[0]["algorithm"] == "fedavg"
    assert local_metrics[0]["train_loss"] == pytest.approx(grpc_metrics[0]["train_loss"])
    assert local_metrics[0]["val_mse"] == pytest.approx(grpc_metrics[0]["val_mse"])
    assert local_metrics[0]["val_mae"] == pytest.approx(grpc_metrics[0]["val_mae"])
    assert local_metrics[0]["val_mape"] == pytest.approx(grpc_metrics[0]["val_mape"])
    assert local_summary["test"]["mse"] == pytest.approx(grpc_summary["test"]["mse"])
    assert local_summary["test"]["mae"] == pytest.approx(grpc_summary["test"]["mae"])
    assert local_summary["test"]["mape"] == pytest.approx(grpc_summary["test"]["mape"])
    assert 'attack_evaluations' not in local_summary
    assert 'attack_evaluations' not in grpc_summary


def test_real_grpc_sync_and_async_match_when_randomness_disabled(tmp_path):
    """The real gRPC transport should preserve deterministic FedAvg artifacts in sync and async modes."""

    sync_dir = tmp_path / "grpc_sync"
    async_dir = tmp_path / "grpc_async"
    port_sync = _free_port()
    port_async = _free_port()
    overrides = [
        "federated.algorithm=fedavg",
        "federated.rounds=2",
        "training.patience=5",
        "attack.enabled=true",
        "attack.target_type=update_payload",
        "replay_capture.frequency_rounds=1",
        "attack.max_samples=1",
        "attack.clients_per_round=1",
        "attack.client_selection=first",
        "attack.steps=1",
        "attack.optimizer=adam",
        "attack.local_optimizer=adam",
        "tracking.enabled=false",
        "runtime.device=cpu",
        "runtime.seed=2026",
        "runtime.deterministic=true",
        "data.shuffle_train=false",
        "model.dropout=0.0",
        "grpc.poll_seconds=0.05",
    ]
    sync_config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(sync_dir),
            f"grpc.address=127.0.0.1:{port_sync}",
            f"grpc.server_address=127.0.0.1:{port_sync}",
            *overrides,
        ],
    )
    async_config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(async_dir),
            f"grpc.address=127.0.0.1:{port_async}",
            f"grpc.server_address=127.0.0.1:{port_async}",
            *overrides,
            "attack.async_enabled=true",
            "attack.async_workers=1",
            "attack.device=cpu",
        ],
    )

    _run_networked_grpc(sync_config)
    _run_networked_grpc(async_config)

    assert compare_fedavg_runs(sync_dir, async_dir, ignore_transport=True) == []


def test_networked_grpc_ega_deferred_runtime_startup(tmp_path):
    config = _classification_grpc_config(tmp_path, algorithm='ega_fedavg')
    port = _free_port()
    config['grpc']['address'] = f'127.0.0.1:{port}'
    config['grpc']['server_address'] = f'127.0.0.1:{port}'
    config['ega']['pretrain'] = {
        'epochs': 1,
        'patience': 1,
        'min_delta': 0.0,
        'batch_size': 8,
        'lr': 1e-3,
        'train_groups': 32,
        'val_groups': 16,
        'seed': 2026,
        'device': 'cpu',
    }

    _run_networked_grpc(config)

    summary = json.loads((Path(config['experiment']['output_dir']) / 'summary.json').read_text(encoding='utf-8'))
    assert summary['transport'] == 'grpc'
    assert 'accuracy' in summary['test']
    assert Path(config['ega']['artifact_path']).exists()



def test_run_client_classification_only_needs_local_train_split(tmp_path, monkeypatch):
    split_dir = tmp_path / 'classification_split'
    _prepare_classification_split_dir(split_dir, client_ids=['m1', 'm2', 'm3'], image_shape=(1, 4, 4), num_classes=3)
    for missing_client in ['m2', 'm3']:
        client_dir = split_dir / 'clients' / missing_client
        for path in client_dir.glob('*'):
            path.unlink()
        client_dir.rmdir()

    config = _classification_grpc_config(tmp_path, algorithm='fedavg')
    config['data']['split_dir'] = str(split_dir)
    captured: dict[str, object] = {}

    class _FakeFederatedClient:
        def __init__(self, client_id, train_loader, _config, _device, total_train_samples=None, total_clients=None, allow_ega_pretrain=False):
            captured['client_id'] = client_id
            captured['local_train_samples'] = len(train_loader.dataset)
            captured['initial_total_train_samples'] = total_train_samples
            captured['initial_total_clients'] = total_clients
            captured['allow_ega_pretrain'] = allow_ega_pretrain

        def registration_payload(self):
            payload = {'client_id': captured['client_id'], 'local_train_samples': captured['local_train_samples'], 'scale_mean': None, 'scale_std': None}
            captured['registration_payload'] = payload
            return payload

    class _FakeRpcClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def snapshot_counters(self):
            return {'sent_bytes': 0, 'received_bytes': 0}

        def register_client(self, payload):
            return {'accepted': True, 'registered_clients': 1, 'expected_clients': 3, 'registration_ready': False}

        def get_global(self):
            return {'stop': True}

        def ack_stop(self, payload):
            return payload

    monkeypatch.setattr(grpc_training_module, 'FederatedClient', _FakeFederatedClient)
    monkeypatch.setattr(grpc_training_module, 'FederatedRpcClient', _FakeRpcClient)

    run_client(config, 'm1')

    assert captured['client_id'] == 'm1'
    assert captured['local_train_samples'] == 6
    assert captured['initial_total_train_samples'] == 6
    assert captured['initial_total_clients'] == 1
    assert captured['registration_payload'] == {'client_id': 'm1', 'local_train_samples': 6, 'scale_mean': None, 'scale_std': None}
    assert captured['allow_ega_pretrain'] is False


def test_run_client_forecasting_only_needs_local_train_split(tmp_path, monkeypatch):
    split_dir = tmp_path / 'rare_split'
    for client in ['Nd2O3', 'CeO2', 'La2O3']:
        client_dir = split_dir / 'clients' / client
        client_dir.mkdir(parents=True, exist_ok=True)
        for split, start, length in [('train', 0, 40), ('val', 40, 20), ('test', 60, 20)]:
            dates = [f'2020-01-{(index % 28) + 1:02d}' for index in range(start, start + length)]
            values = torch.arange(start, start + length, dtype=torch.float32).tolist()
            rows = 'date,value\n' + '\n'.join(f'{date},{value}' for date, value in zip(dates, values)) + '\n'
            (client_dir / f'{split}.csv').write_text(rows, encoding='utf-8')
    for missing_client in ['CeO2', 'La2O3']:
        client_dir = split_dir / 'clients' / missing_client
        for path in client_dir.glob('*'):
            path.unlink()
        client_dir.rmdir()

    config = {
        'experiment': {'output_dir': str(tmp_path / 'forecasting_client'), 'mode': 'federated'},
        'runtime': {'device': 'cpu', 'log_level': 'INFO', 'deterministic': True, 'seed': 2026},
        'task': {'type': 'forecasting'},
        'data': {
            'split_dir': str(split_dir),
            'clients': ['Nd2O3', 'CeO2', 'La2O3'],
            'seq_len': 4,
            'pred_len': 2,
            'batch_size': 8,
            'shuffle_train': False,
            'num_workers': 0,
        },
        'model': {'name': 'lstm', 'input_dim': 1, 'hidden_dim': 8, 'num_layers': 1, 'dropout': 0.0, 'pred_len': 2},
        'training': {'epochs': 1, 'lr': 0.001, 'optimizer': 'adam', 'loss': 'mse', 'patience': 1, 'min_delta': 0.0},
        'centralized': {},
        'federated': {'algorithm': 'fedavg', 'rounds': 1},
        'replay_capture': {'enabled': True, 'frequency_rounds': 1},
        'attack': {'enabled': False, 'async_enabled': False, 'device': 'cpu'},
        'tracking': {'enabled': False},
        'evaluation': {'metrics': ['mse', 'mae', 'mape']},
        'artifacts': {'config_formats': ['yaml'], 'save_every_rounds': 0},
        'grpc': {'address': '127.0.0.1:50051', 'server_address': '127.0.0.1:50051', 'poll_seconds': 0.05, 'max_message_mb': 32.0},
    }
    captured: dict[str, object] = {}

    class _FakeFederatedClient:
        def __init__(self, client_id, train_loader, _config, _device, total_train_samples=None, total_clients=None, allow_ega_pretrain=False):
            captured['client_id'] = client_id
            captured['local_train_samples'] = len(train_loader.dataset)
            captured['initial_total_train_samples'] = total_train_samples
            captured['initial_total_clients'] = total_clients
            captured['allow_ega_pretrain'] = allow_ega_pretrain

        def registration_payload(self):
            payload = {'client_id': captured['client_id'], 'local_train_samples': captured['local_train_samples'], 'scale_mean': [19.5], 'scale_std': [11.54339599609375]}
            captured['registration_payload'] = payload
            return payload

    class _FakeRpcClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def snapshot_counters(self):
            return {'sent_bytes': 0, 'received_bytes': 0}

        def register_client(self, payload):
            captured['registration_payload'] = payload
            return {'accepted': True, 'registered_clients': 1, 'expected_clients': 3, 'registration_ready': False}

        def get_global(self):
            return {'stop': True}

        def ack_stop(self, payload):
            return payload

    monkeypatch.setattr(grpc_training_module, 'FederatedClient', _FakeFederatedClient)
    monkeypatch.setattr(grpc_training_module, 'FederatedRpcClient', _FakeRpcClient)

    run_client(config, 'Nd2O3')

    assert captured['client_id'] == 'Nd2O3'
    assert captured['local_train_samples'] == 35
    assert captured['initial_total_train_samples'] == 35
    assert captured['initial_total_clients'] == 1
    assert captured['registration_payload']['client_id'] == 'Nd2O3'
    assert captured['registration_payload']['local_train_samples'] == 35
    assert captured['registration_payload']['scale_mean'] == pytest.approx([19.5])
    assert captured['registration_payload']['scale_std'] == pytest.approx([11.54339599609375])
    assert captured['allow_ega_pretrain'] is False


def test_grpc_finalization_does_not_block_last_submit_response(tmp_path, monkeypatch):
    """The final client submit should return stop promptly even if finalization is slow."""

    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=fedavg",
            "federated.rounds=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    coordinator = GrpcFederatedCoordinator(config)
    train_loaders, _, _ = build_federated_loaders(config)
    _register_loaded_clients(coordinator, train_loaders)
    device = resolve_device(config)
    global_payload = coordinator.get_global()
    finalized = Event()
    original_finalize = coordinator._finalize

    def slow_finalize() -> None:
        time.sleep(0.2)
        original_finalize()
        finalized.set()

    monkeypatch.setattr(coordinator, '_finalize', slow_finalize)

    response = None
    submit_duration = None
    for client_id, loader in train_loaders.items():
        client = FederatedClient(client_id, loader, config, device)
        result = client.train(global_payload["state"], compressed=global_payload["compressed"], round_index=global_payload["round"])
        start = time.perf_counter()
        response = coordinator.submit_update({"round": global_payload["round"], "result": result})
        duration = time.perf_counter() - start
        if response["stop"]:
            submit_duration = duration
            break

    assert response is not None
    assert response["stop"] is True
    assert submit_duration is not None
    assert submit_duration < 0.3
    assert coordinator.finalize_requested is True
    assert coordinator.finalization_completed is False

    coordinator.finalize_if_requested()

    assert finalized.is_set()
    assert coordinator.finalization_completed is True
    assert (tmp_path / "summary.json").exists()



def test_grpc_coordinator_restores_best_validation_checkpoint(tmp_path, monkeypatch):
    """The gRPC coordinator should test and persist the best validation checkpoint, not the last round."""

    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=fedavg",
            "federated.rounds=2",
            "training.patience=10",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    val_loader = object()
    test_loader = object()

    def _build_zero_model(_config):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model

    monkeypatch.setattr(
        server_module,
        "build_model",
        _build_zero_model,
    )
    minimal_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(1, 1), torch.zeros(1, 1)),
        batch_size=1,
        shuffle=False,
    )
    monkeypatch.setattr(
        __import__('fedlab.communication.grpc_training', fromlist=['build_federated_loaders']),
        'build_federated_loaders',
        lambda _config: (
            {"Nd2O3": minimal_loader, "CeO2": minimal_loader, "La2O3": minimal_loader},
            val_loader,
            test_loader,
        ),
    )

    def fake_evaluate(model, loader, device):
        weight = float(model.weight.item())
        if loader is val_loader:
            mse = 0.1 if weight == 1.0 else 0.2
            return {"mse": mse, "mae": weight, "mape": weight}
        if loader is test_loader:
            mse = 10.0 if weight == 1.0 else 20.0
            return {"mse": mse, "mae": mse / 10.0, "mape": mse / 5.0}
        raise AssertionError("unexpected loader")

    monkeypatch.setattr(server_module, "evaluate", fake_evaluate)

    coordinator = GrpcFederatedCoordinator(config)
    _register_loaded_clients(coordinator, {client_id: minimal_loader for client_id in coordinator.expected_clients})
    response = None
    for _round in range(2):
        payload = coordinator.get_global()
        update = type(payload["state"])((name, torch.ones_like(tensor)) for name, tensor in payload["state"].items())
        for client_id in coordinator.expected_clients:
            result = ClientResult(
                client_id=client_id,
                num_samples=1,
                loss=float(_round + 1),
                aggregation_state=update,
                dense_bytes=4,
                dense_parameters=1,
                download_bytes=4,
                download_parameters=1,
                parameter_download_bytes=4,
                parameter_download_parameters=1,
                dense_download_reference_bytes=4,
                dense_download_reference_parameters=1,
                upload_bytes=4,
                upload_parameters=1,
                parameter_upload_bytes=4,
                parameter_upload_parameters=1,
                transport_download_bytes=4,
                transport_upload_bytes=4,
            )
            response = coordinator.submit_update({"round": payload["round"], "result": result})
    assert response is not None and response["stop"] is True
    coordinator.finalize_if_requested()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    saved_state = torch.load(tmp_path / "model.pt", map_location="cpu")

    assert summary["test"]["mse"] == 10.0
    assert summary["best_round"] == 0
    assert summary["best_val_mse"] == 0.1
    assert summary["test_checkpoint"] == "best_validation"
    assert float(saved_state["weight"].item()) == pytest.approx(1.0)
