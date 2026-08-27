#!/usr/bin/env python3
"""Replay only iDLG from saved server-visible update captures."""

from __future__ import annotations

from fedlab.tools.replay_saved_update_common import run_replay_cli


def main() -> None:
    """Run the CLI that replays only iDLG from saved updates."""

    run_replay_cli(description=__doc__ or "", forced_methods=("idlg",))


if __name__ == "__main__":
    main()
