from pathlib import Path

from federated_ts.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_rawdata2_training_lengths_are_consistent():
    """Rawdata2 centralized and federated configs use the agreed 500-step budget."""

    base = load_config(CONFIG_DIR / "rawdata2_patchtst.yaml")
    fedaware = load_config(CONFIG_DIR / "rawdata2_fedaware.yaml")
    secure_quantized = load_config(CONFIG_DIR / "rawdata2_secure_quantized_fedavg.yaml")
    dp_topk = load_config(CONFIG_DIR / "rawdata2_dp_topk.yaml")
    topk = load_config(CONFIG_DIR / "rawdata2_fedlab_topk.yaml")
    soteriafl = load_config(CONFIG_DIR / "rawdata2_soteriafl.yaml")
    randomk = load_config(CONFIG_DIR / "rawdata2_randomk.yaml")
    sign = load_config(CONFIG_DIR / "rawdata2_sign.yaml")
    qsgd = load_config(CONFIG_DIR / "rawdata2_qsgd.yaml")
    ega = load_config(CONFIG_DIR / "rawdata2_ega.yaml")

    assert base["training"]["epochs"] == 500
    for config in (base, fedaware, secure_quantized, dp_topk, topk, soteriafl, randomk, sign, qsgd, ega):
        assert config["federated"]["rounds"] == 500
        assert config["federated"]["local_epochs"] == 1


def test_shared_configs_do_not_define_compression_only_parameters():
    """Shared base configs should not carry compression-specific federated knobs."""

    base = load_config(CONFIG_DIR / "rawdata2_patchtst.yaml")
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
