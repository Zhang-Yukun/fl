from pathlib import Path

from fedlab.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_retained_rare_configs_share_expected_training_schedule():
    configs = [
        load_config(CONFIG_DIR / "rare/centralized.yaml"),
        load_config(CONFIG_DIR / "rare/fedavg.yaml"),
        load_config(CONFIG_DIR / "rare/topk.yaml"),
        load_config(CONFIG_DIR / "rare/ega.yaml"),
    ]

    assert configs[0]["training"]["patience"] == 500
    assert configs[0]["training"]["epochs"] == 300
    for config in configs[1:]:
        assert config["federated"]["rounds"] == 300
        assert config["training"]["epochs"] == 1
        assert "local_epochs" not in config["federated"]


def test_retained_rare_configs_only_define_relevant_algorithm_blocks():
    fedavg = load_config(CONFIG_DIR / "rare/fedavg.yaml")
    topk = load_config(CONFIG_DIR / "rare/topk.yaml")
    ega = load_config(CONFIG_DIR / "rare/ega.yaml")

    assert "topk_fraction" not in fedavg["federated"]
    assert topk["federated"]["topk_fraction"] == 0.10
    assert "ega" not in fedavg
    assert "ega" not in topk
    assert "topk_fraction" not in ega["federated"]
    assert "ega" in ega


def test_retained_rare_configs_default_to_forecasting_protocol_evaluation():
    configs = [
        load_config(CONFIG_DIR / "rare/centralized.yaml"),
        load_config(CONFIG_DIR / "rare/fedavg.yaml"),
        load_config(CONFIG_DIR / "rare/topk.yaml"),
        load_config(CONFIG_DIR / "rare/ega.yaml"),
    ]

    for config in configs:
        assert config["task"]["type"] == "forecasting"
        assert config.get("evaluation", {}).get("mode", "protocol") == "protocol"
        assert config["evaluation"]["metrics"] == ["mse", "mae", "mape"]


def test_centralized_and_federated_rare_bases_keep_separate_epoch_defaults():
    centralized = load_config(CONFIG_DIR / "rare/centralized.yaml")
    fedavg = load_config(CONFIG_DIR / "rare/fedavg.yaml")

    assert centralized["training"]["epochs"] == 300
    assert centralized["federated"]["rounds"] == 20
    assert fedavg["training"]["epochs"] == 1
    assert fedavg["federated"]["rounds"] == 300
