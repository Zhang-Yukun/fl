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
        assert config["model"]["name"] == "patchtst"


def test_centralized_and_federated_rare_bases_keep_separate_epoch_defaults():
    centralized = load_config(CONFIG_DIR / "rare/centralized.yaml")
    fedavg = load_config(CONFIG_DIR / "rare/fedavg.yaml")

    assert centralized["training"]["epochs"] == 300
    assert centralized["federated"]["rounds"] == 20
    assert fedavg["training"]["epochs"] == 1
    assert fedavg["federated"]["rounds"] == 300


def test_retained_rare_ega_config_uses_shared_common_preset_without_stale_keys():
    ega = load_config(CONFIG_DIR / "rare/ega.yaml")

    assert ega["ega"]["artifact_path"] == "artifacts/ega/ega_h240_v1.pt"
    assert ega["ega"]["block_size"] == 256
    assert ega["ega"]["encoded_dim"] == 168
    assert ega["ega"]["hidden_dim"] == 2048
    assert ega["ega"]["residual_blocks"] == 4
    assert ega["ega"]["quantization_level"] == 159
    assert ega["ega"]["normalization_ema"] == 0.98
    assert ega["ega"]["pretrain"]["epochs"] == 220
    assert ega["ega"]["pretrain"]["patience"] == 44
    assert ega["ega"]["pretrain"]["lr"] == 0.0002
    assert ega["ega"]["pretrain"]["train_groups"] == 50000
    assert ega["ega"]["pretrain"]["val_groups"] == 25000
    assert ega["ega"]["pretrain"]["batch_size"] == 128
    assert ega["ega"]["pretrain"]["seed"] == 2026
    assert ega["ega"]["pretrain"]["device"] == "same"
    assert "download_dtype" not in ega["ega"]
    assert "download_method" not in ega["ega"]
    assert "download_predictive_coding" not in ega["ega"]
    assert "download_stochastic_rounding" not in ega["ega"]
    assert "download_trainable_only" not in ega["ega"]
    assert "download_encoded_dtype" not in ega["ega"]


def test_common_ega_directory_keeps_only_the_shared_default_config():
    ega_files = {path.name for path in (CONFIG_DIR / "common" / "ega").glob("*.yaml")}

    assert ega_files == {"default.yaml"}
