import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import federated_ts.federated.algorithms as algorithms_module
import federated_ts.federated.client as client_module
import federated_ts.federated.server as server_module
from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.engine.training import train_n_steps
from federated_ts.federated.client import FederatedClient
from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import compress_topk, serialize_model

from federated_ts.federated.algorithms import (
    AsyncAttackManager,
    AttackRoundResult,
    AttackRoundTask,
    _protect_attack_gradients,
    _round_attack_payload,
    _round_history_communication_summary,
    _wandb_cumulative_communication_payload,
    run_centralized,
    run_federated,
)
from federated_ts.utils.config import load_config
from federated_ts.utils.consistency import compare_fedavg_runs


def test_one_round_federated_run(tmp_path):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    assert result["rounds"] == 1
    assert result["last_upload_compression_ratio"] >= 6.0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics[0]["total_upload_bytes"] > 0
    assert metrics[0]["total_download_bytes"] > 0
    assert metrics[0]["total_parameter_upload_bytes"] == metrics[0]["total_upload_bytes"]
    assert metrics[0]["total_parameter_download_bytes"] == metrics[0]["total_download_bytes"]
    assert metrics[0]["total_transport_upload_bytes"] >= metrics[0]["total_parameter_upload_bytes"]
    assert metrics[0]["total_transport_download_bytes"] >= metrics[0]["total_parameter_download_bytes"]
    assert metrics[0]["model_parameters"] > 0
    assert {client["client_id"] for client in metrics[0]["clients"]} == {"Nd2O3", "CeO2", "La2O3"}
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "config.yaml").exists()
    assert not (tmp_path / "config.json").exists()


def test_federated_run_saves_config_before_training_starts(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)

    def fail_build(_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(algorithms_module, "build_federated_loaders", fail_build)

    with pytest.raises(RuntimeError, match="boom"):
        run_federated(config)

    assert (tmp_path / "config.yaml").exists()


def test_centralized_run_saves_config_before_training_starts(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)

    def fail_build(_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(algorithms_module, "build_federated_loaders", fail_build)

    with pytest.raises(RuntimeError, match="boom"):
        run_centralized(config)

    assert (tmp_path / "config.yaml").exists()


def test_wandb_cumulative_communication_payload_uses_history_totals(tmp_path):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.algorithm=fedavg"])
    config["experiment"]["output_dir"] = str(tmp_path)
    run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    summary = _round_history_communication_summary([])

    assert summary["total_parameter_bytes"] == 0

    from federated_ts.federated.server import RoundRecord

    record = RoundRecord(**metrics[0])
    payload = _wandb_cumulative_communication_payload([record])

    assert payload["cumulative/total_parameter_upload_bytes"] == metrics[0]["total_parameter_upload_bytes"]
    assert payload["cumulative/total_parameter_download_bytes"] == metrics[0]["total_parameter_download_bytes"]
    assert payload["cumulative/total_transport_bytes"] == metrics[0]["total_transport_bytes"]


def test_protect_attack_gradients_keeps_fedavg_dense_signal():
    config = {"federated": {"algorithm": "fedavg"}, "attack": {"seed": 7}}
    grads = [torch.tensor([[1.0, -2.0]]), torch.tensor([3.0])]

    protected = _protect_attack_gradients(config, grads, round_index=0, client_index=0, sample_index=0)

    assert len(protected) == len(grads)
    assert all(torch.equal(left, right) for left, right in zip(protected, grads))


def test_protect_attack_gradients_applies_dp_topk_mask():
    config = {
        "federated": {"algorithm": "dp_topk_fedavg", "topk_fraction": 0.25},
        "privacy": {"clip_norm": 100.0, "noise_multiplier": 0.0},
        "attack": {"seed": 11},
    }
    grads = [torch.tensor([1.0, -3.0, 2.0, 0.5]), torch.tensor([0.1, -0.2, 5.0, 0.3])]

    protected = _protect_attack_gradients(config, grads, round_index=0, client_index=0, sample_index=0)
    flat = torch.cat([tensor.reshape(-1) for tensor in protected])

    assert torch.count_nonzero(flat).item() == 2
    assert flat[1].item() == -3.0
    assert flat[6].item() == 5.0


def test_standard_fedavg_uses_dense_updates(tmp_path):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.algorithm=fedavg"])
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    assert result["last_upload_compression_ratio"] == 1.0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert all(client["payload_kind"] == "dense_update" for client in metrics[0]["clients"])
    assert metrics[0]["total_upload_bytes"] == metrics[0]["fedavg_reference_upload_bytes"]
    assert metrics[0]["total_parameter_upload_bytes"] == metrics[0]["fedavg_reference_upload_bytes"]
    assert metrics[0]["transport_upload_compression_ratio"] == 1.0


def test_fedaware_uses_dense_updates_and_records_weights(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedaware",
            "federated.rounds=1",
            "attack.enabled=false",
            "fedaware.alpha=1.0",
            "fedaware.steps=10",
            "fedaware.lr=0.2",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]
    assert result["last_upload_compression_ratio"] == 1.0
    assert all(client["payload_kind"] == "dense_update" for client in clients)
    assert abs(sum(client["aggregation_weight"] for client in clients) - 1.0) < 1e-6
    assert all(client["aggregation_weight"] >= 0.0 for client in clients)


def test_config_artifact_formats_are_configurable(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        ["artifacts.config_formats=[yaml,json,toml]"],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    run_federated(config)
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "config.toml").exists()


def test_federated_run_saves_periodic_snapshot(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.rounds=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "artifacts.save_every_rounds=1",
        ],
    )
    run_federated(config)

    snapshot_dir = tmp_path / "snapshots" / "round_0001"
    assert (snapshot_dir / "config.yaml").exists()
    assert (snapshot_dir / "metrics.json").exists()
    assert (snapshot_dir / "summary.json").exists()
    assert (snapshot_dir / "model.pt").exists()
    assert (snapshot_dir / "resume_state.pt").exists()


def test_federated_run_saves_attack_results_for_update_payloads(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "attack.enabled=true",
            "attack.target_type=update_payload",
            "attack.frequency_rounds=1",
            "attack.max_samples=1",
            "attack.steps=1",
            "attack.optimizer=adam",
            "attack.local_optimizer=adam",
            "federated.algorithm=fedavg",
            "federated.rounds=1",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    attack_results = json.loads((tmp_path / "attack_results.json").read_text(encoding="utf-8"))
    assert {entry["name"] for entry in attack_results} == {"DLG", "iDLG"}
    assert {entry["target_type"] for entry in attack_results} == {"update_payload"}
    assert {"mse", "reconstruction_mse", "psnr", "ssim", "iterations", "time_seconds", "objective_mse", "exact_target_mse", "nearest_client_train_mse", "metric_name"} <= set(attack_results[0])
    assert result["attack_evaluations"] == 6
    assert result["attack_target_type"] == "update_payload"
    assert result["attack_primary_metric"] == "nearest_client_train_mse"
    assert result["attack_primary_metric_direction"] == "higher_is_more_private"
    assert result["attack_overall_avg_mse"] is not None
    assert set(result["attack_summary"]["methods"]) == {"DLG", "iDLG"}
    assert result["attack_summary"]["primary_metric"] == "nearest_client_train_mse"
    assert result["attack_summary"]["target_type"] == "update_payload"
    assert result["attack_summary"]["success_rate_threshold"] == 0.03
    assert result["attack_summary"]["overall_avg_nearest_client_train_mse"] is not None
    assert result["attack_summary"]["methods"]["DLG"]["total_count"] == 3


def test_federated_run_supports_legacy_gradient_attacks(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "attack.enabled=true",
            "attack.target_type=gradient",
            "attack.frequency_rounds=1",
            "attack.max_samples=1",
            "attack.steps=1",
            "federated.algorithm=fedavg",
            "federated.rounds=1",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    attack_results = json.loads((tmp_path / "attack_results.json").read_text(encoding="utf-8"))
    assert {entry["target_type"] for entry in attack_results} == {"gradient"}
    assert result["attack_target_type"] == "gradient"


def test_attack_task_uses_protocol_payload_not_oracle_evaluation_update():
    config = {
        "federated": {"algorithm": "compressed_fedavg"},
        "attack": {
            "enabled": True,
            "target_type": "update_payload",
            "frequency_rounds": 1,
            "max_samples": 1,
            "sample_count": 1,
            "client_selection": "all",
        },
    }
    protocol_update = serialize_model(torch.nn.Linear(2, 1, bias=False))
    protocol_update["weight"] = torch.tensor([[0.0, 2.0]])
    oracle_update = serialize_model(torch.nn.Linear(2, 1, bias=False))
    oracle_update["weight"] = torch.tensor([[9.0, 9.0]])
    result = client_module.ClientResult(
        client_id="Nd2O3",
        num_samples=1,
        loss=0.0,
        sparse_update=compress_topk(protocol_update, 0.5),
        evaluation_update=oracle_update,
        payload_kind="sparse_update",
    )
    client = SimpleNamespace(
        client_id="Nd2O3",
        sample_batch=lambda max_samples=None, batch_index=0: (torch.zeros(1, 2, 1), torch.zeros(1, 1, 1)),
        train_reference_inputs=lambda: torch.zeros(1, 2, 1),
    )

    task = algorithms_module._build_attack_round_task(
        config,
        [client],
        [result],
        round_index=0,
        max_rounds=1,
        round_base_state=serialize_model(torch.nn.Linear(2, 1, bias=False)),
        attack_target_type="update_payload",
    )

    assert task is not None
    target = task.samples[0].target
    assert isinstance(target, dict)
    assert torch.equal(target["weight"], torch.tensor([[0.0, 2.0]]))
    assert not torch.equal(target["weight"], oracle_update["weight"])


def test_soteriafl_uses_sparse_dp_payloads(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=soteriafl",
            "federated.rounds=1",
            "privacy.clip_norm=1.0",
            "privacy.noise_multiplier=0.0",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert result["last_upload_compression_ratio"] >= 6.0
    assert all(client["payload_kind"] == "soteriafl_randomk_dp_update" for client in metrics[0]["clients"])
    assert all(client["compressor"] == "randomk_unbiased" for client in metrics[0]["clients"])
    assert all(client["privacy_clip_norm"] == 1.0 for client in metrics[0]["clients"])



def test_secure_quantized_fedavg_uses_quantized_dense_updates(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=secure_quantized_fedavg",
            "federated.rounds=1",
            "federated.quantization_dtype=float16",
            "privacy.clip_norm=10.0",
            "privacy.noise_multiplier=0.0",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]

    assert result["last_upload_compression_ratio"] > 1.5
    assert result["last_total_communication_ratio"] > 1.9
    assert all(client["payload_kind"] == "quantized_update" for client in clients)
    assert all(client["compressor"] == "float16_quantized_dense" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)
    assert all(client["download_bytes"] < client["dense_download_reference_bytes"] for client in clients)
    assert all(client["parameter_upload_bytes"] == client["upload_bytes"] for client in clients)
    assert all(client["parameter_download_bytes"] == client["download_bytes"] for client in clients)


def test_secure_quantized_fedavg_supports_absmax_int8(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=secure_quantized_fedavg",
            "federated.rounds=1",
            "federated.quantization_dtype=int8",
            "federated.quantization_stochastic_rounding=true",
            "federated.quantization_seed=2026",
            "privacy.clip_norm=10.0",
            "privacy.noise_multiplier=0.0",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]

    assert result["last_upload_compression_ratio"] > 3.0
    assert result["last_total_communication_ratio"] > 3.0
    assert all(client["compressor"] == "int8_quantized_dense" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)
    assert all(client["download_bytes"] < client["dense_download_reference_bytes"] for client in clients)
    assert all(client["parameter_upload_bytes"] == client["upload_bytes"] for client in clients)
    assert all(client["parameter_download_bytes"] == client["download_bytes"] for client in clients)


def test_oracle_evaluation_separates_protocol_metrics_from_dense_reference_updates(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.algorithm=compressed_fedavg",
            "federated.rounds=1",
            "federated.topk_fraction=0.5",
            "evaluation.mode=oracle_full_update",
            "attack.enabled=false",
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

    monkeypatch.setattr(algorithms_module, "build_model", _build_linear)
    monkeypatch.setattr(server_module, "build_model", _build_linear)
    monkeypatch.setattr(
        algorithms_module,
        "build_federated_loaders",
        lambda _config: ({"Nd2O3": _StaticLoader(), "CeO2": _StaticLoader(), "La2O3": _StaticLoader()}, val_loader, test_loader),
    )

    dense_update = serialize_model(_build_linear(config))
    dense_update["weight"] = torch.tensor([[1.0, 2.0]])

    def fake_train(self, global_state, compressed=False, round_index=0):
        return client_module.ClientResult(
            client_id=self.client_id,
            num_samples=1,
            loss=0.0,
            sparse_update=compress_topk(dense_update, 0.5),
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

    def fake_evaluate(model, loader, device):
        weight = model.weight.detach().cpu().clone()
        first = float(weight[0, 0].item())
        second = float(weight[0, 1].item())
        mse = (1.0 - first) ** 2 + (2.0 - second) ** 2
        return {"mse": mse, "mae": abs(1.0 - first) + abs(2.0 - second), "mape": mse}

    monkeypatch.setattr(client_module.FederatedClient, "train", fake_train)
    monkeypatch.setattr(algorithms_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(server_module, "evaluate", fake_evaluate)

    summary = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert summary["evaluation_mode"] == "oracle_full_update"
    assert summary["test"]["mse"] == pytest.approx(0.0)
    assert summary["protocol_test"]["mse"] > 0.0
    assert summary["oracle_test"]["mse"] == pytest.approx(0.0)
    assert metrics[0]["protocol_val_mse"] > 0.0
    assert metrics[0]["oracle_val_mse"] == pytest.approx(0.0)
    assert (tmp_path / "oracle_model.pt").exists()


def test_dp_topk_uses_sparse_dp_topk_payloads(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=dp_topk_fedavg",
            "federated.rounds=1",
            "privacy.clip_norm=1.0",
            "privacy.noise_multiplier=0.0",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert result["last_upload_compression_ratio"] >= 6.0
    assert all(client["payload_kind"] == "dp_topk_dp_update" for client in metrics[0]["clients"])
    assert all(client["compressor"] == "topk_dp" for client in metrics[0]["clients"])
    assert all(client["privacy_clip_norm"] == 1.0 for client in metrics[0]["clients"])


def test_train_n_steps_cycles_loader_and_validates_steps():
    device = torch.device("cpu")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [(torch.ones(2, 1), torch.zeros(2, 1))]

    with pytest.raises(ValueError):
        train_n_steps(model, loader, optimizer, device, steps=0)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_n_steps(model, loader, optimizer, device, steps=3)
    after = list(model.parameters())

    assert loss >= 0.0
    assert any(not torch.allclose(left, right.detach()) for left, right in zip(before, after))


def test_client_prefers_local_steps_over_local_epochs(monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "federated.local_steps=2",
            "federated.local_epochs=7",
            "attack.enabled=false",
        ],
    )
    config["runtime"]["device"] = "cpu"
    train_loaders, _, _ = build_federated_loaders(config)
    client_id, loader = next(iter(train_loaders.items()))
    client = FederatedClient(client_id, loader, config, torch.device("cpu"))
    global_state = serialize_model(build_model(config))
    calls = {"steps": 0, "epochs": 0}

    def fake_train_n_steps(model, loader, optimizer, device, steps):
        calls["steps"] += 1
        assert steps == 2
        return 1.25

    def fake_train_one_epoch(model, loader, optimizer, device):
        calls["epochs"] += 1
        return 9.0

    monkeypatch.setattr(client_module, "train_n_steps", fake_train_n_steps)
    monkeypatch.setattr(client_module, "train_one_epoch", fake_train_one_epoch)

    result = client.train(global_state, compressed=False, round_index=0)

    assert calls == {"steps": 1, "epochs": 0}
    assert result.loss == 1.25


def test_select_attack_clients_defaults_to_all_clients():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", [])
    clients = [
        SimpleNamespace(client_id="Nd2O3"),
        SimpleNamespace(client_id="CeO2"),
        SimpleNamespace(client_id="La2O3"),
    ]

    selected = algorithms_module._select_attack_clients(clients, config, round_index=7)

    assert [client.client_id for client in selected] == ["Nd2O3", "CeO2", "La2O3"]


def test_async_attacks_match_sync_fedavg_when_randomness_disabled(tmp_path):
    sync_dir = tmp_path / "sync"
    async_dir = tmp_path / "async"
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
        "tracking.enabled=false",
        "runtime.device=cpu",
        "runtime.seed=2026",
        "runtime.deterministic=true",
        "data.shuffle_train=false",
        "model.dropout=0.0",
    ]
    sync_config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["experiment.output_dir=" + str(sync_dir), *overrides])
    async_config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(async_dir),
            *overrides,
            "attack.async_enabled=true",
            "attack.async_workers=1",
            "attack.async_device=cpu",
        ],
    )

    sync_summary = run_federated(sync_config)
    async_summary = run_federated(async_config)

    sync_artifacts = sorted((sync_dir / "attack_artifacts").rglob("*.pt"))
    async_artifacts = sorted((async_dir / "attack_artifacts").rglob("*.pt"))

    assert sync_artifacts
    assert async_artifacts
    assert sync_summary["test"]["mse"] == pytest.approx(async_summary["test"]["mse"])
    assert sync_summary["attack_overall_avg_mse"] == pytest.approx(async_summary["attack_overall_avg_mse"])
    assert sync_summary["attack_success_rate"] == pytest.approx(async_summary["attack_success_rate"])
    assert compare_fedavg_runs(sync_dir, async_dir) == []


class _TrackerStub:
    def __init__(self):
        self.logs = []

    def log(self, data, step=None):
        self.logs.append((step, data))


def _attack_result_stub(name: str, mse: float = 0.5):
    return SimpleNamespace(
        name=name,
        mse=mse,
        psnr=10.0,
        ssim=0.1,
        iterations=1,
        time_seconds=0.01,
        gradient_mse=0.02,
        success=False,
    )


def test_async_attack_manager_preserves_sync_mode(monkeypatch):
    tracker = _TrackerStub()
    config = {"attack": {"enabled": True, "async_enabled": False}}
    task = AttackRoundTask(round_index=0, clients_this_round=1, samples_per_client=1, samples=[])

    def fake_execute(config, task, attack_device):
        return AttackRoundResult(
            round_index=task.round_index,
            time_seconds=0.1,
            clients_this_round=task.clients_this_round,
            samples_per_client=task.samples_per_client,
            attacks=[_attack_result_stub("DLG")],
        )

    monkeypatch.setattr(algorithms_module, "_execute_attack_round_task", fake_execute)

    manager = AsyncAttackManager(config, tracker)
    manager.submit(task)

    assert manager.executor is None
    assert len(manager.attack_results) == 1
    assert tracker.logs[0][0] == 0


def test_async_attack_manager_applies_pending_round_backpressure(monkeypatch):
    tracker = _TrackerStub()
    config = {
        "attack": {
            "enabled": True,
            "async_enabled": True,
            "async_workers": 1,
            "async_max_pending_rounds": 1,
        }
    }

    class FakeFuture:
        def __init__(self, result):
            self._result = result
            self._done = False

        def done(self):
            return self._done

        def result(self):
            self._done = True
            return self._result

    futures = [
        FakeFuture(AttackRoundResult(0, 0.1, 1, 1, [_attack_result_stub("DLG", mse=0.4)])),
        FakeFuture(AttackRoundResult(1, 0.1, 1, 1, [_attack_result_stub("DLG", mse=0.6)])),
    ]

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            self.submit_calls = 0

        def submit(self, fn, config, task, attack_device):
            future = futures[self.submit_calls]
            self.submit_calls += 1
            return future

        def shutdown(self, wait=True):
            return None

    def fake_wait(pending, return_when=None):
        futures[0]._done = True
        return {futures[0]}, set()

    monkeypatch.setattr(algorithms_module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(algorithms_module, "wait", fake_wait)

    manager = AsyncAttackManager(config, tracker)
    manager.submit(AttackRoundTask(round_index=0, clients_this_round=1, samples_per_client=1, samples=[]))
    assert len(manager.pending_round_order) == 1
    manager.submit(AttackRoundTask(round_index=1, clients_this_round=1, samples_per_client=1, samples=[]))

    assert tracker.logs[0][0] == 0
    assert manager.pending_round_order == [1]
    manager.finalize()
    assert [step for step, _ in tracker.logs] == [0, 1]
    assert len(manager.attack_results) == 2


def test_round_attack_payload_includes_explicit_round_index():
    round_result = AttackRoundResult(
        round_index=3,
        time_seconds=0.2,
        clients_this_round=1,
        samples_per_client=1,
        attacks=[_attack_result_stub("DLG", mse=0.4)],
    )

    payload = _round_attack_payload(round_result, round_result.attacks)

    assert payload["attack/round_index"] == 3.0


class _DummyLoader:
    def __init__(self, size: int = 1):
        self.dataset = [0] * size



def test_centralized_run_restores_best_validation_checkpoint(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "training.epochs=2",
            "training.patience=10",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    val_loader = object()
    test_loader = object()
    model = torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(
        algorithms_module,
        "build_federated_loaders",
        lambda _config: ({"client_a": _DummyLoader()}, val_loader, test_loader),
    )
    monkeypatch.setattr(algorithms_module, "build_model", lambda _config: model)

    epoch_state = {"count": 0}

    def fake_train_one_epoch(model, loader, optimizer, device):
        epoch_state["count"] += 1
        with torch.no_grad():
            model.weight.fill_(float(epoch_state["count"]))
        return float(epoch_state["count"])

    def fake_evaluate(model, loader, device):
        weight = float(model.weight.item())
        if loader is val_loader:
            mse = 0.1 if weight == 1.0 else 0.2
            return {"mse": mse, "mae": weight, "mape": weight}
        if loader is test_loader:
            mse = 10.0 if weight == 1.0 else 20.0
            return {"mse": mse, "mae": mse / 10.0, "mape": mse / 5.0}
        raise AssertionError("unexpected loader")

    monkeypatch.setattr(algorithms_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(algorithms_module, "evaluate", fake_evaluate)

    result = run_centralized(config)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    saved_state = torch.load(tmp_path / "centralized_model.pt", map_location="cpu")

    assert result["mse"] == 10.0
    assert summary["best_epoch"] == 0
    assert summary["best_val_mse"] == 0.1
    assert summary["test_checkpoint"] == "best_validation"
    assert metrics["best_epoch"] == 0
    assert metrics["best_val"]["mse"] == 0.1
    assert float(saved_state["weight"].item()) == pytest.approx(1.0)



def test_federated_run_restores_best_validation_checkpoint(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "federated.rounds=2",
            "training.patience=10",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    val_loader = object()
    test_loader = object()

    monkeypatch.setattr(
        algorithms_module,
        "build_federated_loaders",
        lambda _config: (
            {"Nd2O3": _DummyLoader(), "CeO2": _DummyLoader(), "La2O3": _DummyLoader()},
            val_loader,
            test_loader,
        ),
    )
    def _build_zero_model(_config):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model

    monkeypatch.setattr(server_module, "build_model", _build_zero_model)

    def fake_client_train(self, global_state, compressed=False, round_index=0):
        update = type(global_state)((name, torch.ones_like(tensor)) for name, tensor in global_state.items())
        return client_module.ClientResult(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            loss=float(round_index + 1),
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

    def fake_evaluate(model, loader, device):
        weight = float(model.weight.item())
        if loader is val_loader:
            mse = 0.1 if weight == 1.0 else 0.2
            return {"mse": mse, "mae": weight, "mape": weight}
        if loader is test_loader:
            mse = 10.0 if weight == 1.0 else 20.0
            return {"mse": mse, "mae": mse / 10.0, "mape": mse / 5.0}
        raise AssertionError("unexpected loader")

    monkeypatch.setattr(client_module.FederatedClient, "train", fake_client_train)
    monkeypatch.setattr(server_module, "evaluate", fake_evaluate)

    summary = run_federated(config)
    saved_state = torch.load(tmp_path / "model.pt", map_location="cpu")
    persisted_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert summary["test"]["mse"] == 10.0
    assert persisted_summary["best_round"] == 0
    assert persisted_summary["best_val_mse"] == 0.1
    assert persisted_summary["test_checkpoint"] == "best_validation"
    assert float(saved_state["weight"].item()) == pytest.approx(1.0)
