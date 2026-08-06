import json
from pathlib import Path

import pytest
import torch

import federated_ts.federated.client as client_module
from federated_ts.datasets.rare_earth import build_federated_loaders
from federated_ts.engine.training import train_n_steps
from federated_ts.federated.client import FederatedClient
from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import serialize_model

from federated_ts.federated.algorithms import (
    _protect_attack_gradients,
    _round_history_communication_summary,
    _wandb_cumulative_communication_payload,
    run_federated,
)
from federated_ts.utils.config import load_config


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
    assert result["attack_evaluations"] == 2
    assert result["attack_target_type"] == "update_payload"
    assert result["attack_primary_metric"] == "nearest_client_train_mse"
    assert result["attack_primary_metric_direction"] == "higher_is_more_private"
    assert result["attack_overall_avg_mse"] is not None
    assert set(result["attack_summary"]["methods"]) == {"DLG", "iDLG"}
    assert result["attack_summary"]["primary_metric"] == "nearest_client_train_mse"
    assert result["attack_summary"]["target_type"] == "update_payload"
    assert result["attack_summary"]["success_rate_threshold"] == 0.03
    assert result["attack_summary"]["overall_avg_nearest_client_train_mse"] is not None
    assert result["attack_summary"]["methods"]["DLG"]["total_count"] == 1


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


def test_fedpetuning_uploads_only_trainable_subset(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedpetuning",
            "federated.rounds=1",
            "attack.enabled=false",
            "model.name=patchtst",
            "model.channels=1",
            "model.patch_len=4",
            "model.stride=2",
            "model.d_model=8",
            "model.n_heads=1",
            "model.e_layers=1",
            "model.d_ff=16",
            "model.peft.enabled=true",
            "model.peft.method=fedpetuning",
            "model.peft.bottleneck_dim=4",
            "model.peft.train_head=true",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]
    assert all(client["payload_kind"] == "fedpetuning_trainable_state" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)
    assert all(client["parameter_upload_bytes"] == client["upload_bytes"] for client in clients)
    assert all(client["transport_upload_bytes"] >= client["parameter_upload_bytes"] for client in clients)
    assert result["last_upload_compression_ratio"] > 1.0


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
