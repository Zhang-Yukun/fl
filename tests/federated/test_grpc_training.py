import json
import multiprocessing
import socket
import time
from pathlib import Path

import pytest

from federated_ts.communication.grpc_training import GrpcFederatedCoordinator, _apply_transport_metrics, run_client, serve
from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.federated.algorithms import resolve_device, run_federated
from federated_ts.federated.client import ClientResult, FederatedClient
from federated_ts.utils.config import load_config
from federated_ts.utils.consistency import compare_fedavg_runs


def _submit_one_round(coordinator: GrpcFederatedCoordinator, config: dict[str, object]) -> dict[str, object]:
    """Submit one full gRPC round using in-process clients for tests."""

    train_loaders, _, _ = build_federated_loaders(config)
    device = resolve_device(config)
    global_payload = coordinator.get_global()
    response = None
    for client_id, loader in train_loaders.items():
        client = FederatedClient(client_id, loader, config, device)
        result = client.train(global_payload["state"], compressed=global_payload["compressed"], round_index=global_payload["round"])
        response = coordinator.submit_update({"round": global_payload["round"], "result": result})
    assert response is not None
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
    assert metrics[0]["total_transport_upload_bytes"] == metrics[0]["total_parameter_upload_bytes"]
    assert summary["transport"] == "grpc"
    assert summary["rounds"] == 1
    assert summary["last_upload_compression_ratio"] == 1.0
    assert summary["total_parameter_bytes"] == metrics[0]["total_parameter_bytes"]
    assert summary["total_transport_bytes"] == metrics[0]["total_transport_bytes"]


def test_grpc_coordinator_saves_attack_results(tmp_path):
    """The gRPC coordinator should persist DLG/iDLG artifacts when attacks are enabled."""

    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=fedavg",
            "federated.rounds=1",
            "attack.enabled=true",
            "attack.target_type=update_payload",
            "attack.frequency_rounds=1",
            "attack.max_samples=1",
            "attack.sample_count=1",
            "attack.clients_per_round=1",
            "attack.steps=1",
            "attack.optimizer=adam",
            "attack.local_optimizer=adam",
            "attack.async_enabled=true",
            "attack.async_workers=1",
            "attack.async_device=cpu",
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
    attack_results = json.loads((tmp_path / "attack_results.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert {entry["name"] for entry in attack_results} == {"DLG", "iDLG"}
    assert {entry["target_type"] for entry in attack_results} == {"update_payload"}
    assert summary["attack_evaluations"] == 2
    assert summary["attack_target_type"] == "update_payload"
    assert summary["attack_primary_metric"] == "nearest_client_train_mse"


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
        "attack.frequency_rounds=1",
        "attack.max_samples=1",
        "attack.sample_count=1",
        "attack.clients_per_round=1",
        "attack.client_selection=first",
        "attack.steps=1",
        "attack.optimizer=adam",
        "attack.local_optimizer=adam",
        "attack.async_enabled=true",
        "attack.async_workers=1",
        "attack.async_device=cpu",
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
    local_attacks = json.loads((local_dir / "attack_results.json").read_text(encoding="utf-8"))
    grpc_attacks = json.loads((grpc_dir / "attack_results.json").read_text(encoding="utf-8"))
    grpc_summary = json.loads((grpc_dir / "summary.json").read_text(encoding="utf-8"))

    assert local_metrics[0]["algorithm"] == grpc_metrics[0]["algorithm"] == "fedavg"
    assert local_metrics[0]["train_loss"] == pytest.approx(grpc_metrics[0]["train_loss"])
    assert local_metrics[0]["val_mse"] == pytest.approx(grpc_metrics[0]["val_mse"])
    assert local_metrics[0]["val_mae"] == pytest.approx(grpc_metrics[0]["val_mae"])
    assert local_metrics[0]["val_mape"] == pytest.approx(grpc_metrics[0]["val_mape"])
    assert local_summary["test"]["mse"] == pytest.approx(grpc_summary["test"]["mse"])
    assert local_summary["test"]["mae"] == pytest.approx(grpc_summary["test"]["mae"])
    assert local_summary["test"]["mape"] == pytest.approx(grpc_summary["test"]["mape"])
    assert local_summary["attack_overall_avg_mse"] == pytest.approx(grpc_summary["attack_overall_avg_mse"])
    assert local_summary["attack_success_rate"] == pytest.approx(grpc_summary["attack_success_rate"])
    assert [entry["name"] for entry in local_attacks] == [entry["name"] for entry in grpc_attacks]
    for local_entry, grpc_entry in zip(local_attacks, grpc_attacks):
        assert local_entry["target_type"] == grpc_entry["target_type"]
        assert local_entry["mse"] == pytest.approx(grpc_entry["mse"])
        assert local_entry["objective_mse"] == pytest.approx(grpc_entry["objective_mse"])


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
        "attack.frequency_rounds=1",
        "attack.max_samples=1",
        "attack.sample_count=1",
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
            "attack.async_device=cpu",
        ],
    )

    _run_networked_grpc(sync_config)
    _run_networked_grpc(async_config)

    assert compare_fedavg_runs(sync_dir, async_dir, ignore_transport=True) == []
