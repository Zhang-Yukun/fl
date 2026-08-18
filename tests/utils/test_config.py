from pathlib import Path

from fedlab.utils.config import load_config


def test_nested_yaml_and_override():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.rounds=2", "tracking.enabled=false"])
    assert config["federated"]["rounds"] == 2
    assert config["data"]["seq_len"] == 21
    assert config["tracking"]["enabled"] is False




def test_default_centralized_epochs_are_mode_specific():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["centralized.epochs=3", "tracking.enabled=false"])
    assert config["centralized"]["epochs"] == 3
    assert config["federated"]["rounds"] == 1
