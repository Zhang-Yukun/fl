"""Experiment tracking with optional wandb integration."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from fedlab.datasets.rare_earth import inverse_transform_tensor


class Tracker:
    """Thin wrapper that keeps training usable when wandb is unavailable."""

    def __init__(self, config: dict[str, Any]):
        """Start a wandb run when tracking is enabled in config."""

        self.run = None
        self.wandb = None
        tracking_cfg = config.get("tracking", {})
        if not tracking_cfg.get("enabled", False):
            return
        try:
            import wandb

            if tracking_cfg.get("offline", True):
                os.environ["WANDB_MODE"] = "offline"
            self.wandb = wandb
            init_kwargs = {"project": tracking_cfg.get("project", "federated-rare-earth"), "config": config}
            if tracking_cfg.get("name"):
                init_kwargs["name"] = str(tracking_cfg.get("name"))
            if tracking_cfg.get("group"):
                init_kwargs["group"] = str(tracking_cfg.get("group"))
            if tracking_cfg.get("job_type"):
                init_kwargs["job_type"] = str(tracking_cfg.get("job_type"))
            if tracking_cfg.get("tags"):
                init_kwargs["tags"] = list(tracking_cfg.get("tags"))
            self.run = wandb.init(**init_kwargs)
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

    def log_image(self, key: str, image: Any, step: int | None = None, caption: str | None = None) -> None:
        """Log one image artifact when wandb is active."""

        if self.run is None or self.wandb is None:
            return
        payload = {key: self.wandb.Image(image, caption=caption)}
        self.log(payload, step=step)
        if hasattr(image, "savefig"):
            try:
                import matplotlib.pyplot as plt

                plt.close(image)
            except Exception:
                pass

    def log_prediction_plot(
        self,
        key: str,
        input_series,
        prediction,
        target,
        step: int | None = None,
        title: str | None = None,
        scaler = None,
    ) -> None:
        """Render and log an input-context plus target-vs-prediction chart."""

        figure = _prediction_figure(input_series, prediction, target, title=title, scaler=scaler)
        if figure is None:
            return
        self.log_image(key, figure, step=step, caption=title)

    def log_attack_reconstruction(self, key: str, result: Any, step: int | None = None) -> None:
        """Render and log one attack reconstruction figure."""

        figure = _attack_reconstruction_figure(result)
        if figure is None:
            return
        self.log_image(key, figure, step=step, caption=f"{getattr(result, 'name', 'attack')} round={getattr(result, 'round_index', 'na')}")

    def finish(self) -> None:
        """Close the wandb run when one was created."""

        if self.run is not None:
            self.run.finish()


def _to_series(tensor) -> list[float]:
    """Flatten a time-series tensor to one plottable 1D sequence.

    Example:
        ``[1, 96, 1]`` becomes a 96-step series instead of collapsing to one point.
    """

    if tensor is None:
        return []
    import torch

    data = tensor.detach().cpu()
    if data.ndim == 0:
        return [float(data.item())]
    if data.ndim == 1:
        return [float(value) for value in data.tolist()]
    if data.ndim >= 2 and data.shape[0] == 1:
        data = data[0]
    while data.ndim > 1 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim > 1:
        data = data.reshape(data.shape[0], -1)[:, 0]
    return [float(value) for value in data.tolist()]


def _prediction_figure(input_series, prediction, target, title: str | None = None, scaler=None):
    """Return a single-axis matplotlib figure for context and forecast comparison."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        logger.warning("Prediction plot disabled: {}", exc)
        return None

    input_values = _to_series(inverse_transform_tensor(input_series, scaler))
    pred_series = _to_series(inverse_transform_tensor(prediction, scaler))
    target_series = _to_series(inverse_transform_tensor(target, scaler))
    if not pred_series or not target_series:
        return None
    figure, axis = plt.subplots(figsize=(10, 3.5))
    history_steps = list(range(len(input_values)))
    forecast_steps = list(range(len(input_values), len(input_values) + len(target_series)))
    if input_values:
        axis.plot(history_steps, input_values, label="input_x", linewidth=2.0)
        if target_series:
            axis.plot([history_steps[-1], forecast_steps[0]], [input_values[-1], target_series[0]], linewidth=1.2, alpha=0.5, color='tab:orange', linestyle=':')
        if pred_series:
            axis.plot([history_steps[-1], forecast_steps[0]], [input_values[-1], pred_series[0]], linewidth=1.2, alpha=0.5, color='tab:green', linestyle=':')
    axis.plot(forecast_steps, target_series, label="target_y", linewidth=2.0)
    axis.plot(forecast_steps, pred_series, label="prediction_y", linewidth=2.0)
    if input_values:
        axis.axvline(len(input_values) - 0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    axis.set_title(title or "Prediction vs target")
    axis.set_xlabel("step")
    axis.set_ylabel("value")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def _attack_reconstruction_figure(result):
    """Return a matplotlib figure for attack reconstruction diagnostics."""

    real_x = getattr(result, "plot_real_x", getattr(result, "real_x", None))
    reconstructed_x = getattr(result, "plot_reconstructed_x", getattr(result, "reconstructed_x", None))
    if real_x is None or reconstructed_x is None:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        logger.warning("Attack plot disabled: {}", exc)
        return None

    real_y = getattr(result, "plot_real_y", getattr(result, "real_y", None))
    reconstructed_y = getattr(result, "plot_reconstructed_y", getattr(result, "reconstructed_y", None))
    figure, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].plot(_to_series(real_x), label="real_x", linewidth=2.0)
    axes[0].plot(_to_series(reconstructed_x), label="reconstructed_x", linewidth=2.0)
    axes[0].set_title("Input reconstruction")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    y_real_series = _to_series(real_y)
    y_recon_series = _to_series(reconstructed_y)
    if y_real_series:
        axes[1].plot(y_real_series, label="real_y", linewidth=2.0)
    if y_recon_series:
        axes[1].plot(y_recon_series, label="reconstructed_y", linewidth=2.0)
    axes[1].set_title("Target reconstruction")
    axes[1].grid(True, alpha=0.3)
    if y_real_series or y_recon_series:
        axes[1].legend(loc="best")
    figure.tight_layout()
    return figure
