#!/usr/bin/env python3
"""Evaluate attack-only reconstruction artifacts against client training data."""

from fedlab.attack.evaluation import run_evaluation_cli


def main() -> None:
    """Run the standalone saved-attack evaluation command."""

    run_evaluation_cli()


if __name__ == "__main__":
    main()
