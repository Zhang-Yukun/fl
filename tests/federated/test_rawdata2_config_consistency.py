from pathlib import Path

from federated_ts.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_rawdata2_training_lengths_are_consistent():
    """Rawdata2 centralized and federated configs use the agreed 500-step budget."""

    base = load_config(CONFIG_DIR / "rawdata2_patchtst.yaml")
    fedaware = load_config(CONFIG_DIR / "rawdata2_fedaware.yaml")
    topk = load_config(CONFIG_DIR / "rawdata2_fedlab_topk.yaml")
    soteriafl = load_config(CONFIG_DIR / "rawdata2_soteriafl.yaml")

    assert base["training"]["epochs"] == 500
    for config in (base, fedaware, topk, soteriafl):
        assert config["federated"]["rounds"] == 500
        assert config["federated"]["local_epochs"] == 1
