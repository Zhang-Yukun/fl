"""Loguru setup helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(output_dir: str | Path, level: str = "INFO") -> None:
    """Configure console and file logging for an experiment run."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level)
    logger.add(output_dir / "run.log", level=level, rotation="10 MB", retention=5)

