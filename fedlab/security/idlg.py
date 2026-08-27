"""Dedicated iDLG attack entrypoint."""

from __future__ import annotations

from typing import Any

import torch

from fedlab.security.attack_common import AttackResult, run_attack_loop
from fedlab.utils.serialization import StateDict


def idlg_attack(
    config: dict[str, Any],
    state: StateDict,
    target: StateDict,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
    target_type: str | None = None,
    reference_inputs: torch.Tensor | None = None,
    reference_targets: torch.Tensor | None = None,
) -> AttackResult:
    """Run iDLG-style reconstruction with label inference when available."""

    return run_attack_loop(
        config,
        state,
        target,
        real_x,
        real_y,
        device,
        name="iDLG",
        optimize_y=False,
        target_type=target_type,
        reference_inputs=reference_inputs,
        reference_targets=reference_targets,
    )
