"""Command-line entry points."""

from __future__ import annotations

import argparse
import json

from fedlab.federated.algorithms import run_centralized, run_federated
from fedlab.utils.config import load_config


def main() -> None:
    """Parse command-line arguments and launch the requested training mode.

    Example:
        ``python -m fedlab.entrypoints.train --config configs/rare/fedavg.yaml``.
    """

    parser = argparse.ArgumentParser(description="Federated rare-earth price forecasting")
    parser.add_argument("--config", default="configs/rare/fedavg.yaml", help="Path to YAML config")
    parser.add_argument("--mode", choices=["federated", "centralized"], default=None)
    parser.add_argument("--override", action="append", default=[], help="Override config value, e.g. federated.rounds=3")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    mode = args.mode or config.get("experiment", {}).get("mode", "federated")
    result = run_centralized(config) if mode == "centralized" else run_federated(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

