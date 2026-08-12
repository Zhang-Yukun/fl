"""Compare two deterministic FedAvg run directories and report non-timing differences."""

from __future__ import annotations

import argparse
import sys

from fedlab.utils.consistency import compare_fedavg_runs


def main() -> int:
    """Compare two experiment directories and print any mismatches."""

    parser = argparse.ArgumentParser(description="Compare deterministic FedAvg run artifacts")
    parser.add_argument("reference_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--ignore-transport", action="store_true")
    args = parser.parse_args()

    diffs = compare_fedavg_runs(
        args.reference_dir,
        args.candidate_dir,
        tolerance=args.tolerance,
        ignore_transport=args.ignore_transport,
    )
    if diffs:
        print("Deterministic consistency check failed:")
        for diff in diffs:
            print(f"- {diff}")
        return 1
    print("Deterministic consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
