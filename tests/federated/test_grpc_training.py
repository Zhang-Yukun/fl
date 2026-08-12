import json
import multiprocessing
import socket
import time
from threading import Event
from pathlib import Path

import pytest
import torch

import federated_ts.federated.server as server_module
import federated_ts.communication.grpc_training as grpc_training_module
import federated_ts.security.attacks as attacks_module
from federated_ts.communication.grpc_training import GrpcFederatedCoordinator, _apply_transport_metrics, run_client, serve
from federated_ts.datasets.rare_earth import build_federated_loaders
import federated_ts.federated.algorithms as algorithms_module
from federated_ts.federated.algorithms import resolve_device, run_federated
from federated_ts.federated.client import ClientResult, FederatedClient
from federated_ts.utils.config import load_config
from federated_ts.utils.serialization import compress_topk, serialize_model
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
            "attack.enabled=false",
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
    assert metrics[0]["total_transport_upload_bytes"] == metrics[0]["total_parameter_upload_bytes"]
    assert summary["transport"] == "grpc"
    assert summary["rounds"] == 1
    assert summary["last_upload_compression_ratio"] == 1.0
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
    artifacts = sorted((tmp_path / "attack_artifacts").rglob("*.pt"))
    assert artifacts
    assert {entry["name"] for entry in attack_results} == {"DLG", "iDLG"}
    assert {entry["target_type"] for entry in attack_results} == {"update_payload"}
    assert summary["attack_evaluations"] == len(attack_results) == 6
    assert summary["attack_target_type"] == "update_payload"
    assert summary["attack_primary_metric"] == "nearest_client_train_mse"


def test_grpc_coordinator_keeps_oracle_evaluation_out_of_attack_payload(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=compressed_fedavg",
            "federated.rounds=1",
            "federated.topk_fraction=0.5",
            "evaluation.mode=oracle_full_update",
            "attack.enabled=true",
            "attack.target_type=update_payload",
            "attack.frequency_rounds=1",
            "attack.sample_count=1",
            "attack.max_samples=1",
            "attack.async_enabled=true",
            "attack.async_workers=1",
            "attack.async_device=cpu",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    val_loader = object()
    test_loader = object()

    def _build_linear(_config):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model

    class _StaticLoader:
        def __init__(self):
            self.dataset = [(torch.zeros(1, 2, 1), torch.zeros(1, 1, 1))]

        def __iter__(self):
            return iter(self.dataset)

    monkeypatch.setattr(server_module, "build_model", _build_linear)
    monkeypatch.setattr(attacks_module, "build_model", _build_linear)
    monkeypatch.setattr(grpc_training_module, "build_federated_loaders", lambda _config: (
        {"Nd2O3": _StaticLoader(), "CeO2": _StaticLoader(), "La2O3": _StaticLoader()},
        val_loader,
        test_loader,
    ))

    dense_update = serialize_model(_build_linear(config))
    dense_update["weight"] = torch.tensor([[1.0, 2.0]])
    sparse_update = compress_topk(dense_update, 0.5)

    def fake_train(self, global_state, compressed=False, round_index=0):
        return ClientResult(
            client_id=self.client_id,
            num_samples=1,
            loss=0.0,
            sparse_update=sparse_update,
            evaluation_update=dense_update,
            dense_bytes=8,
            dense_parameters=2,
            download_bytes=8,
            download_parameters=2,
            parameter_download_bytes=8,
            parameter_download_parameters=2,
            dense_download_reference_bytes=8,
            dense_download_reference_parameters=2,
            upload_bytes=4,
            upload_parameters=1,
            parameter_upload_bytes=4,
            parameter_upload_parameters=1,
            transport_download_bytes=8,
            transport_upload_bytes=4,
            payload_kind="sparse_update",
            compressor="topk",
        )

    original_build_attack_round_task = grpc_training_module._build_attack_round_task
    captured_targets = []

    def fake_build_attack_round_task(*args, **kwargs):
        task = original_build_attack_round_task(*args, **kwargs)
        if task is not None:
            sample_target = task.samples[0].target
            captured_targets.append(sample_target["weight"].detach().cpu().clone())
        return task

    def fake_evaluate(model, loader, device):
        weight = model.weight.detach().cpu().clone()
        first = float(weight[0, 0].item())
        second = float(weight[0, 1].item())
        mse = (1.0 - first) ** 2 + (2.0 - second) ** 2
        return {"mse": mse, "mae": abs(1.0 - first) + abs(2.0 - second), "mape": mse}

    monkeypatch.setattr(FederatedClient, "train", fake_train)
    monkeypatch.setattr(grpc_training_module, "_build_attack_round_task", fake_build_attack_round_task)
    monkeypatch.setattr(server_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        algorithms_module,
        "_execute_attack_round_task",
        lambda config, task, attack_device: algorithms_module.AttackRoundResult(
            round_index=task.round_index,
            time_seconds=0.0,
            clients_this_round=task.clients_this_round,
            samples_per_client=task.samples_per_client,
            attacks=[attacks_module.AttackResult(name="DLG", mse=1.0, psnr=0.0, ssim=0.0, iterations=1, time_seconds=0.0, success=False, success_threshold=0.03, gradient_mse=0.0, target_type="update_payload")],
        ),
    )

    coordinator = GrpcFederatedCoordinator(config)
    response = _submit_one_round(coordinator, config)

    assert response["stop"] is True
    assert captured_targets
    assert torch.equal(captured_targets[0], torch.tensor([[0.0, 2.0]]))
    assert not torch.equal(captured_targets[0], dense_update["weight"])

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluation_mode"] == "oracle_full_update"
    assert summary["protocol_test"]["mse"] > 0.0
    assert summary["oracle_test"]["mse"] == pytest.approx(0.0)


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

    assert sorted((local_dir / "attack_artifacts").rglob("*.pt"))
    assert sorted((grpc_dir / "attack_artifacts").rglob("*.pt"))
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
    assert submit_duration < 0.15
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
    monkeypatch.setattr(
        __import__('federated_ts.communication.grpc_training', fromlist=['build_federated_loaders']),
        'build_federated_loaders',
        lambda _config: (
            {"Nd2O3": [0], "CeO2": [0], "La2O3": [0]},
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
    response = None
    for _round in range(2):
        payload = coordinator.get_global()
        update = type(payload["state"])((name, torch.ones_like(tensor)) for name, tensor in payload["state"].items())
        for client_id in coordinator.expected_clients:
            result = ClientResult(
                client_id=client_id,
                num_samples=1,
                loss=float(_round + 1),
                state=update,
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
