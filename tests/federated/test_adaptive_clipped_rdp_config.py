from pathlib import Path

from fedlab.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_rawdata2_adaptive_clipped_rdp_config_matches_formal_round_budget():
    config = load_config(CONFIG_DIR / "rawdata2_adaptive_clipped_rdp_fedavg.yaml")

    assert config["federated"]["algorithm"] == "adaptive_clipped_rdp_fedavg"
    assert config["federated"]["rounds"] == 300
    assert config["federated"]["local_epochs"] == 1
    assert config["adaptive_clipped_rdp"]["reference_clip_norm"] > 0
    assert config["adaptive_clipped_rdp"]["rdp_alpha"] > 1
