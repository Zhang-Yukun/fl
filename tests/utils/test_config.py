from pathlib import Path

from federated_ts.utils.config import load_config


def test_nested_yaml_and_override():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.rounds=2", "tracking.enabled=false"])
    assert config["federated"]["rounds"] == 2
    assert config["data"]["seq_len"] == 21
    assert config["tracking"]["enabled"] is False

