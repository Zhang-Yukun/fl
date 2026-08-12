import json
import tomllib

import torch
import yaml

from federated_ts.utils.artifacts import normalize_config_formats, save_experiment_config, save_federated_snapshot, should_save_periodic_artifacts


def test_normalize_config_formats_defaults_and_deduplicates():
    assert normalize_config_formats(None) == ["yaml"]
    assert normalize_config_formats("yaml,json,yaml") == ["yaml", "json"]


def test_save_experiment_config_multiple_formats(tmp_path):
    config = {
        "experiment": {"name": "demo"},
        "runtime": {"device": "cpu"},
        "data": {"clients": ["Nd2O3", "CeO2", "La2O3"]},
    }
    paths = save_experiment_config(config, tmp_path, ["yaml", "json", "toml"])
    assert [path.name for path in paths] == ["config.yaml", "config.json", "config.toml"]
    assert yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) == config
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == config
    assert tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))["experiment"]["name"] == "demo"



def test_should_save_periodic_artifacts_uses_round_interval():
    assert should_save_periodic_artifacts({"artifacts": {"save_every_rounds": 2}}, 1) is False
    assert should_save_periodic_artifacts({"artifacts": {"save_every_rounds": 2}}, 2) is True
    assert should_save_periodic_artifacts({"artifacts": {"save_every_rounds": 0}}, 4) is False


def test_save_federated_snapshot_writes_expected_files(tmp_path):
    config = {"artifacts": {"config_formats": ["yaml"]}}
    snapshot_dir = save_federated_snapshot(
        tmp_path,
        config,
        snapshot_name="round_0002",
        model_state={"weight": torch.tensor([1.0])},
        oracle_model_state={"weight": torch.tensor([2.0])},
        metrics_history=[{"round": 1, "val_mse": 0.1}],
        summary={"rounds": 1, "test": {"mse": 0.2}},
        attack_records=[{"name": "DLG", "mse": 1.0}],
        resume_state={"round_index": 2},
    )

    assert snapshot_dir == tmp_path / "snapshots" / "round_0002"
    assert (snapshot_dir / "config.yaml").exists()
    assert (snapshot_dir / "metrics.json").exists()
    assert (snapshot_dir / "summary.json").exists()
    assert (snapshot_dir / "attack_results.json").exists()
    assert (snapshot_dir / "model.pt").exists()
    assert (snapshot_dir / "oracle_model.pt").exists()
    assert (snapshot_dir / "resume_state.pt").exists()
    assert json.loads((snapshot_dir / "summary.json").read_text(encoding="utf-8"))["rounds"] == 1
