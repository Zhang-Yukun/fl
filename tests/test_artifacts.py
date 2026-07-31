import json
import tomllib

import yaml

from federated_ts.artifacts import normalize_config_formats, save_experiment_config


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
