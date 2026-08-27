"""Start a standalone gRPC federated server process."""

import argparse

from fedlab.utils.config import load_config
from fedlab.communication.grpc_training import serve


def main() -> None:
    """Start the blocking gRPC server from command-line arguments."""

    parser = argparse.ArgumentParser(description="Federated gRPC server")
    parser.add_argument("--config", default="configs/rare/fedavg.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    serve(load_config(args.config, args.override))


if __name__ == "__main__":
    main()
