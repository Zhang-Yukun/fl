"""Run rawdata2 PatchTST centralized and FedAvg experiments."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from federated_ts.federated.algorithms import run_centralized, run_federated
from federated_ts.utils.config import load_config
from scripts.prepare_rawdata2 import prepare_rawdata2


def main() -> None:
    """Prepare data, then run centralized and standard FedAvg PatchTST jobs."""

    parser = argparse.ArgumentParser(description="Run rawdata2 PatchTST experiments")
    parser.add_argument("--config", default="configs/rawdata2_patchtst.yaml")
    parser.add_argument("--prepare-data", action="store_true", default=False)
    parser.add_argument("--raw-dir", default="../Time-Series-Prediction/dataset/data_preprocess/rawdata2")
    parser.add_argument("--data-dir", default="../data/rare_earth_rawdata2")
    args = parser.parse_args()
    if args.prepare_data:
        prepare_rawdata2(Path(args.raw_dir), Path(args.data_dir))
    base = load_config(args.config)
    centralized = deepcopy(base)
    centralized["experiment"]["mode"] = "centralized"
    centralized["experiment"]["output_dir"] = "outputs/rawdata2_patchtst_centralized"
    fedavg = deepcopy(base)
    fedavg["experiment"]["mode"] = "federated"
    fedavg["experiment"]["output_dir"] = "outputs/rawdata2_patchtst_fedavg"
    fedavg["federated"]["algorithm"] = "fedavg"
    results = {"centralized": run_centralized(centralized), "fedavg": run_federated(fedavg)}
    summary_path = Path("outputs/rawdata2_patchtst_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
