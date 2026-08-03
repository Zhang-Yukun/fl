import torch
import json
from pathlib import Path

from federated_ts.federated.algorithms import _protect_attack_gradients, run_federated
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
    assert metrics[0]["model_parameters"] > 0
    assert {client["client_id"] for client in metrics[0]["clients"]} == {"Nd2O3", "CeO2", "La2O3"}
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "config.yaml").exists()
    assert not (tmp_path / "config.json").exists()


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


def test_standard_fedavg_uses_dense_uploads(tmp_path):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.algorithm=fedavg"])
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    assert result["last_upload_compression_ratio"] == 1.0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert all(client["payload_kind"] == "dense_state" for client in metrics[0]["clients"])
    assert metrics[0]["total_upload_bytes"] == metrics[0]["fedavg_reference_upload_bytes"]


def test_fedaware_uses_dense_uploads_and_records_weights(tmp_path):
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
    assert all(client["payload_kind"] == "dense_state" for client in clients)
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


def test_federated_run_saves_attack_results(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "attack.enabled=true",
            "attack.frequency_rounds=1",
            "attack.max_samples=1",
            "attack.steps=1",
            "federated.rounds=1",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    attack_results = json.loads((tmp_path / "attack_results.json").read_text(encoding="utf-8"))
    assert {entry["name"] for entry in attack_results} == {"DLG", "iDLG"}
    assert {"mse", "reconstruction_mse", "psnr", "ssim", "iterations", "time_seconds"} <= set(attack_results[0])
    assert result["attack_evaluations"] == 2
    assert set(result["attack_summary"]["methods"]) == {"DLG", "iDLG"}
    assert result["attack_summary"]["success_rate_threshold"] == 0.03
    assert result["attack_summary"]["methods"]["DLG"]["total_count"] == 1


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
    assert result["last_upload_compression_ratio"] > 1.0


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
