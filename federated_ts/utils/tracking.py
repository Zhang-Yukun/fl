"""Experiment tracking with optional wandb integration."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


class Tracker:
    """Thin wrapper that keeps training usable when wandb is unavailable."""

    def __init__(self, config: dict[str, Any]):
        """Start a wandb run when tracking is enabled in config."""

        self.run = None
        tracking_cfg = config.get("tracking", {})
        if not tracking_cfg.get("enabled", False):
            return
        try:
            import wandb

            if tracking_cfg.get("offline", True):
                os.environ["WANDB_MODE"] = "offline"
            self.run = wandb.init(project=tracking_cfg.get("project", "federated-rare-earth"), config=config)
        except Exception as exc:  # pragma: no cover - depends on optional environment
            logger.warning("wandb disabled: {}", exc)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb when tracking is enabled.

        Example:
            ``tracker.log({"round/loss": 0.1}, step=3)`` also records
            ``tracking/step=3`` so asynchronous producers still carry an
            explicit round index in the payload itself.
        """

        payload = dict(data)
        if step is not None:
            payload.setdefault("tracking/step", int(step))
        if self.run is not None:
            self.run.log(payload, step=step)

    def finish(self) -> None:
        """Close the wandb run when one was created."""

        if self.run is not None:
            self.run.finish()

