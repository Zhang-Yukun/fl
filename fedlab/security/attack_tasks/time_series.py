"""Time-series-specific helpers for reconstruction attacks."""

from __future__ import annotations

import torch


def time_series_total_variation(x: torch.Tensor) -> torch.Tensor:
    """Return a small total-variation regularizer for 1D windows."""

    if x.ndim < 3 or x.shape[1] <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))


def idlg_uses_dlg_target_optimization() -> bool:
    """Return whether time-series iDLG should fall back to DLG target optimization."""

    return True
