from pathlib import Path

from fedlab.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_training_lengths_are_consistent():
    """Retained algorithm configs share the agreed training/round budgets."""

    configs = [
        load_config(CONFIG_DIR / "fedavg.yaml"),
        load_config(CONFIG_DIR / "topk.yaml"),
        load_config(CONFIG_DIR / "qsgd.yaml"),
        load_config(CONFIG_DIR / "randomk.yaml"),
        load_config(CONFIG_DIR / "sign.yaml"),
        load_config(CONFIG_DIR / "adaptive_clipped_rdp_fedavg.yaml"),
        load_config(CONFIG_DIR / "secure_quantized_fedavg.yaml"),
        load_config(CONFIG_DIR / "ega.yaml"),
    ]

    assert configs[0]["training"]["patience"] == 301
    for config in configs:
        assert config["federated"]["rounds"] == 300
        assert config["federated"]["local_epochs"] == 1


def test_shared_configs_do_not_define_compression_only_parameters():
    """Shared base configs should not carry compression-specific federated knobs."""

    base = load_config(CONFIG_DIR / "fedavg.yaml")
    default = load_config(CONFIG_DIR / "default.yaml")
    topk = load_config(CONFIG_DIR / "topk.yaml")
    randomk = load_config(CONFIG_DIR / "randomk.yaml")

    assert "topk_fraction" not in base["federated"]
    assert "topk_fraction" not in default["federated"]
    assert topk["federated"]["topk_fraction"] == 0.10
    assert randomk["federated"]["topk_fraction"] == 0.10


def test_all_algorithm_configs_default_to_protocol_evaluation():
    """All algorithm configs default to protocol evaluation unless explicitly overridden."""

    configs = [
        load_config(CONFIG_DIR / "fedavg.yaml"),
        load_config(CONFIG_DIR / "adaptive_clipped_rdp_fedavg.yaml"),
        load_config(CONFIG_DIR / "ega.yaml"),
        load_config(CONFIG_DIR / "topk.yaml"),
        load_config(CONFIG_DIR / "qsgd.yaml"),
        load_config(CONFIG_DIR / "randomk.yaml"),
        load_config(CONFIG_DIR / "secure_quantized_fedavg.yaml"),
        load_config(CONFIG_DIR / "sign.yaml"),
    ]

    for config in configs:
        assert config.get("evaluation", {}).get("mode", "protocol") == "protocol"


def test_centralized_rounds_do_not_override_federated_rounds():
    centralized = load_config(CONFIG_DIR / "centralized.yaml")
    fedavg = load_config(CONFIG_DIR / "fedavg.yaml")

    assert centralized.get("centralized", {}).get("rounds") == 500
    assert "epochs" not in centralized.get("training", {})
    assert fedavg.get("centralized", {}).get("rounds") == 10
    assert fedavg["federated"]["rounds"] == 300


def test_algorithm_configs_do_not_carry_unrelated_blocks():
    fedavg = load_config(CONFIG_DIR / "fedavg.yaml")
    secure_quantized = load_config(CONFIG_DIR / "secure_quantized_fedavg.yaml")
    adaptive = load_config(CONFIG_DIR / "adaptive_clipped_rdp_fedavg.yaml")
    ega = load_config(CONFIG_DIR / "ega.yaml")

    assert "privacy" not in fedavg
    assert "privacy" in secure_quantized
    assert "adaptive_clipped_rdp" not in secure_quantized
    assert "ega" not in secure_quantized
    assert "privacy" not in adaptive
    assert "adaptive_clipped_rdp" in adaptive
    assert "privacy" not in ega
    assert "ega" in ega
