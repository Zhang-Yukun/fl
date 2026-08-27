#!/usr/bin/env python3
"""Replay saved server-visible update attacks with the configured attack order."""

from __future__ import annotations

from fedlab.tools.replay_saved_update_common import run_replay_cli


def main() -> None:
    """Run the CLI that replays the configured saved-update attacks."""

    run_replay_cli(description=__doc__ or "")


if __name__ == "__main__":
    main()
