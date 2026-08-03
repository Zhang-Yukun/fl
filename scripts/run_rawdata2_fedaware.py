"""Run rawdata2 PatchTST with Xu et al.'s FedAWARE aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from federated_ts.federated.algorithms import run_federated
from federated_ts.utils.config import load_config
from scripts.prepare_rawdata2 import prepare_rawdata2


def main() -> None:
    """Prepare data optionally, then run the configured FedAWARE experiment.

    Example:
        ``python -m scripts.run_rawdata2_fedaware --override federated.rounds=2``
        runs a short Xu et al. adaptive aggregation smoke test.
    """

    parser = argparse.ArgumentParser(description="Run rawdata2 FedAWARE PatchTST experiment")
    parser.add_argument("--config", default="configs/rawdata2_fedaware.yaml")
    parser.add_argument("--prepare-data", action="store_true", default=False)
    parser.add_argument("--raw-dir", default="../Time-Series-Prediction/dataset/data_preprocess/rawdata2")
    parser.add_argument("--data-dir", default="../data/rare_earth_rawdata2")
    parser.add_argument("--override", action="append", default=[], help="Override config value, e.g. federated.rounds=2")
    args = parser.parse_args()
    if args.prepare_data:
        prepare_rawdata2(Path(args.raw_dir), Path(args.data_dir))
    config = load_config(args.config, args.override)
    config["experiment"]["mode"] = "federated"
    config["federated"]["algorithm"] = "fedaware"
    result = run_federated(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
