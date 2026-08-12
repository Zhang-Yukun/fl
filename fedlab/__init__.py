"""Federated time-series learning framework for rare-earth price prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    """Lazily import the config loader so utility modules do not pull optional deps."""

    from fedlab.utils.config import load_config as _load_config

    return _load_config(path, overrides)


__all__ = ["load_config"]
