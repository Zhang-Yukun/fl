from pathlib import Path

from fedlab.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_rawdata2_training_lengths_are_consistent():
    """Rawdata2 configs share the agreed rawdata2 training/round budgets."""

    base = load_config(CONFIG_DIR / "rawdata2_fedavg.yaml")
    fedaware = load_config(CONFIG_DIR / "rawdata2_fedaware.yaml")
    secure_quantized = load_config(CONFIG_DIR / "rawdata2_secure_quantized_fedavg.yaml")
    dp_topk = load_config(CONFIG_DIR / "rawdata2_dp_topk.yaml")
    topk = load_config(CONFIG_DIR / "rawdata2_fedlab_topk.yaml")
    soteriafl = load_config(CONFIG_DIR / "rawdata2_soteriafl.yaml")
    randomk = load_config(CONFIG_DIR / "rawdata2_randomk.yaml")
    sign = load_config(CONFIG_DIR / "rawdata2_sign.yaml")
    qsgd = load_config(CONFIG_DIR / "rawdata2_qsgd.yaml")
    ega = load_config(CONFIG_DIR / "rawdata2_ega.yaml")

    assert base["training"]["patience"] == 301
    for config in (base, fedaware, secure_quantized, dp_topk, topk, soteriafl, randomk, sign, qsgd, ega):
        assert config["federated"]["rounds"] == 300
        assert config["federated"]["local_epochs"] == 1


def test_shared_configs_do_not_define_compression_only_parameters():
    """Shared base configs should not carry compression-specific federated knobs."""

    base = load_config(CONFIG_DIR / "rawdata2_fedavg.yaml")
    default = load_config(CONFIG_DIR / "default.yaml")
    dp_topk = load_config(CONFIG_DIR / "rawdata2_dp_topk.yaml")
    topk = load_config(CONFIG_DIR / "rawdata2_fedlab_topk.yaml")
    soteriafl = load_config(CONFIG_DIR / "rawdata2_soteriafl.yaml")
    randomk = load_config(CONFIG_DIR / "rawdata2_randomk.yaml")

    assert "topk_fraction" not in base["federated"]
    assert "topk_fraction" not in default["federated"]
    assert dp_topk["federated"]["topk_fraction"] == 0.50
    assert topk["federated"]["topk_fraction"] == 0.10
    assert soteriafl["federated"]["topk_fraction"] == 0.10
    assert randomk["federated"]["topk_fraction"] == 0.10


def test_oracle_evaluation_enabled_only_for_approximate_methods():
    """Approximate/perturbed rawdata2 methods should enable oracle_full_update evaluation."""

    fedavg = load_config(CONFIG_DIR / "rawdata2_fedavg.yaml")
    fedaware = load_config(CONFIG_DIR / "rawdata2_fedaware.yaml")
    approximate = [
        load_config(CONFIG_DIR / "rawdata2_adaptive_clipped_rdp_fedavg.yaml"),
        load_config(CONFIG_DIR / "rawdata2_dp_topk.yaml"),
        load_config(CONFIG_DIR / "rawdata2_ega.yaml"),
        load_config(CONFIG_DIR / "rawdata2_fedlab_topk.yaml"),
        load_config(CONFIG_DIR / "rawdata2_qsgd.yaml"),
        load_config(CONFIG_DIR / "rawdata2_randomk.yaml"),
        load_config(CONFIG_DIR / "rawdata2_secure_quantized_fedavg.yaml"),
        load_config(CONFIG_DIR / "rawdata2_sign.yaml"),
        load_config(CONFIG_DIR / "rawdata2_soteriafl.yaml"),
    ]

    assert fedavg.get("evaluation", {}).get("mode", "protocol") == "protocol"
    assert fedaware.get("evaluation", {}).get("mode", "protocol") == "protocol"
    for config in approximate:
        assert config.get("evaluation", {}).get("mode") == "oracle_full_update"


def test_centralized_rounds_are_not_shared_with_federated_configs():
    centralized = load_config(CONFIG_DIR / "rawdata2_centralized.yaml")
    fedavg = load_config(CONFIG_DIR / "rawdata2_fedavg.yaml")

    assert centralized.get("centralized", {}).get("rounds") == 500
    assert "epochs" not in centralized.get("training", {})
    assert fedavg.get("centralized", {}).get("rounds") is None
    assert fedavg["federated"]["rounds"] == 300


def test_algorithm_configs_do_not_carry_unrelated_blocks():
    fedavg = load_config(CONFIG_DIR / "rawdata2_fedavg.yaml")
    secure_quantized = load_config(CONFIG_DIR / "rawdata2_secure_quantized_fedavg.yaml")
    adaptive = load_config(CONFIG_DIR / "rawdata2_adaptive_clipped_rdp_fedavg.yaml")
    ega = load_config(CONFIG_DIR / "rawdata2_ega.yaml")

    assert "privacy" not in fedavg
    assert "fedaware" not in secure_quantized
    assert "adaptive_clipped_rdp" not in secure_quantized
    assert "ega" not in secure_quantized
    assert "privacy" not in adaptive
    assert "adaptive_clipped_rdp" in adaptive
    assert "privacy" not in ega
    assert "ega" in ega
