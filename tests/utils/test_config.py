from pathlib import Path

import pytest

from fedlab.utils.config import load_config


def test_nested_yaml_and_override():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.rounds=2", "tracking.enabled=false"])
    assert config["federated"]["rounds"] == 2
    assert config["data"]["seq_len"] == 21
    assert config["tracking"]["enabled"] is False




def test_default_centralized_rounds_are_mode_specific():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["centralized.rounds=3", "tracking.enabled=false"])
    assert config["centralized"]["rounds"] == 3
    assert config["federated"]["rounds"] == 1


def test_load_config_prunes_irrelevant_algorithm_specific_fields():
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "federated.topk_fraction=0.25",
            "federated.quantization_dtype=int8",
            "fedaware.alpha=0.7",
            "adaptive_clipped_rdp.noise_multiplier=0.5",
            "ega.block_size=64",
            "privacy.clip_norm=2.0",
        ],
    )
    assert "topk_fraction" not in config["federated"]
    assert "quantization_dtype" not in config["federated"]
    assert "fedaware" not in config
    assert "adaptive_clipped_rdp" not in config
    assert "ega" not in config
    assert "privacy" not in config


def test_load_config_keeps_only_active_algorithm_specific_fields():
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=secure_quantized_fedavg",
            "federated.topk_fraction=0.25",
            "federated.quantization_dtype=int8",
            "federated.quantization_seed=2026",
            "privacy.clip_norm=2.0",
            "privacy.noise_multiplier=0.1",
            "fedaware.alpha=0.7",
        ],
    )
    assert config["federated"]["quantization_dtype"] == "int8"
    assert config["federated"]["quantization_seed"] == 2026
    assert "topk_fraction" not in config["federated"]
    assert config["privacy"]["clip_norm"] == 2.0
    assert "fedaware" not in config


def test_deprecated_centralized_epochs_key_is_rejected():
    with pytest.raises(ValueError, match=r"centralized\.epochs"):
        load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["centralized.epochs=3"])


def test_deprecated_training_epochs_key_is_rejected():
    with pytest.raises(ValueError, match=r"training\.epochs"):
        load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["training.epochs=3"])
