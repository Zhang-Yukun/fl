"""Thin compatibility wrapper around the standalone attack replay module."""

from __future__ import annotations

from fedlab.attack.replay import (
    build_replay_config,
    build_replay_summary,
    default_config_path,
    replay_saved_update_attacks,
    round_indices,
    run_replay_cli,
)

__all__ = [
    "default_config_path",
    "round_indices",
    "build_replay_config",
    "build_replay_summary",
    "replay_saved_update_attacks",
    "run_replay_cli",
]
