from pathlib import Path

from fedlab.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_adaptive_clipped_rdp_config_can_be_materialized_from_test_base():
    config = load_config(
        CONFIG_DIR / "test.yaml",
        [
            "federated.algorithm=adaptive_clipped_rdp_fedavg",
            "federated.rounds=300",
            "adaptive_clipped_rdp.noise_multiplier=0.001",
            "adaptive_clipped_rdp.reference_clip_norm=10.0",
            "adaptive_clipped_rdp.min_clip_norm=0.1",
            "adaptive_clipped_rdp.max_clip_norm=10.0",
            "adaptive_clipped_rdp.clip_factor=1.2",
            "adaptive_clipped_rdp.rdp_alpha=16.0",
            "adaptive_clipped_rdp.delta=1.0e-5",
            "adaptive_clipped_rdp.total_clients=3",
            "adaptive_clipped_rdp.seed=2026",
        ],
    )

    assert config["federated"]["algorithm"] == "adaptive_clipped_rdp_fedavg"
    assert config["federated"]["rounds"] == 300
    assert config["training"]["epochs"] == 1
    assert "local_epochs" not in config["federated"]
    assert config["adaptive_clipped_rdp"]["reference_clip_norm"] > 0
    assert config["adaptive_clipped_rdp"]["rdp_alpha"] > 1
