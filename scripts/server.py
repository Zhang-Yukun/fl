"""Start a standalone gRPC federated server process."""

import argparse

from federated_ts.config import load_config
from federated_ts.grpc_training import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated gRPC server")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    serve(load_config(args.config, args.override))


if __name__ == "__main__":
    main()
