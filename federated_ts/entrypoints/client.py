"""Start a standalone gRPC federated client process."""

import argparse

from federated_ts.utils.config import load_config
from federated_ts.communication.grpc_training import run_client



def main() -> None:
    """Start one gRPC federated client from command-line arguments."""

    parser = argparse.ArgumentParser(description="Federated gRPC client")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run_client(load_config(args.config, args.override), args.client_id)


if __name__ == "__main__":
    main()
