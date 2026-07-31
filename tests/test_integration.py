import json
from pathlib import Path

from federated_ts.algorithms import run_federated
from federated_ts.config import load_config


def test_one_round_federated_run(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs" / "test.yaml")
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


def test_standard_fedavg_uses_dense_uploads(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs" / "test.yaml", ["federated.algorithm=fedavg"])
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    assert result["last_upload_compression_ratio"] == 1.0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert all(client["payload_kind"] == "dense_state" for client in metrics[0]["clients"])
    assert metrics[0]["total_upload_bytes"] == metrics[0]["fedavg_reference_upload_bytes"]


def test_config_artifact_formats_are_configurable(tmp_path):
    config = load_config(
        Path(__file__).parents[1] / "configs" / "test.yaml",
        ["artifacts.config_formats=[yaml,json,toml]"],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    run_federated(config)
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "config.toml").exists()
