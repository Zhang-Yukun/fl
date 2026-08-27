"""Backward-compatible facade for reconstruction attacks."""

from fedlab.security.attack_common import (
    AttackResult,
    apply_set_recovery_metrics,
    attach_attack_metadata,
    attack_success_rate,
    save_attack_artifacts,
    summarize_attack_results,
)
from fedlab.security.dlg import dlg_attack
from fedlab.security.idlg import idlg_attack

__all__ = [
    "AttackResult",
    "apply_set_recovery_metrics",
    "attach_attack_metadata",
    "attack_success_rate",
    "dlg_attack",
    "idlg_attack",
    "save_attack_artifacts",
    "summarize_attack_results",
]
