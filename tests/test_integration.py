from pathlib import Path

from federated_ts.algorithms import run_federated
from federated_ts.config import load_config


def test_one_round_federated_run(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs" / "test.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    assert result["rounds"] == 1
    assert result["last_communication_ratio"] >= 6.0
    assert (tmp_path / "model.pt").exists()

