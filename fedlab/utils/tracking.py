"""Experiment tracking with optional wandb integration."""

from __future__ import annotations

import os
from typing import Any

import torch
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
            if hasattr(image, "savefig"):
                try:
                    import matplotlib.pyplot as plt

                    plt.close(image)
                except Exception:
                    pass
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


def _first_sample_tensor(tensor):
    """Return the first sample from a batch-like tensor on CPU."""

    if tensor is None:
        return None
    data = tensor.detach().cpu()
    if data.ndim >= 4:
        return data[0]
    if data.ndim >= 1 and data.shape[0] == 1 and data.ndim >= 3:
        return data[0]
    return data


def _image_tensor(tensor):
    """Return one image tensor in CHW/HW form when the payload looks image-like."""

    sample = _first_sample_tensor(tensor)
    if sample is None:
        return None
    if sample.ndim == 2 and min(sample.shape) >= 2:
        return sample.to(torch.float32)
    if sample.ndim != 3 or sample.shape[0] not in (1, 3):
        return None
    height, width = int(sample.shape[1]), int(sample.shape[2])
    if min(height, width) < 2:
        return None
    aspect_ratio = max(height, width) / max(1, min(height, width))
    if aspect_ratio > 4.0:
        return None
    return sample.to(torch.float32)


def _render_image(axis, tensor, title: str) -> None:
    """Render one grayscale or RGB tensor on a matplotlib axis."""

    image = _image_tensor(tensor)
    if image is None:
        axis.text(0.5, 0.5, 'image unavailable', ha='center', va='center')
        axis.set_axis_off()
        axis.set_title(title)
        return
    if image.ndim == 2:
        axis.imshow(image.numpy(), cmap='gray')
    elif image.shape[0] == 1:
        axis.imshow(image[0].numpy(), cmap='gray')
    else:
        axis.imshow(image.permute(1, 2, 0).clamp(0.0, 1.0).numpy())
    axis.set_title(title)
    axis.set_axis_off()


def _classification_label_text(tensor) -> str | None:
    """Return a readable class-label summary for one target tensor."""

    if tensor is None:
        return None
    data = tensor.detach().cpu()
    if torch.is_floating_point(data):
        return None
    if data.ndim == 0:
        return str(int(data.item()))
    if data.ndim == 1:
        return ', '.join(str(int(value)) for value in data.tolist())
    if data.ndim >= 2 and data.shape[0] == 1:
        flattened = data.reshape(-1).tolist()
        return ', '.join(str(int(value)) for value in flattened)
    return None


def _classification_probabilities(tensor) -> torch.Tensor | None:
    """Return one probability vector for classification logits/probabilities."""

    if tensor is None:
        return None
    data = tensor.detach().cpu().to(torch.float32)
    if data.ndim == 0:
        return None
    if data.ndim == 1 and data.numel() > 1:
        return torch.softmax(data, dim=0)
    if data.ndim >= 2:
        sample = data.reshape(data.shape[0], -1) if data.ndim > 2 else data
        if sample.shape[-1] > 1:
            return torch.softmax(sample[0], dim=0)
    return None


def _classification_prediction_summary(tensor) -> tuple[int, float] | None:
    """Return the top-1 predicted class and confidence for one classification tensor."""

    probabilities = _classification_probabilities(tensor)
    if probabilities is None or probabilities.numel() == 0:
        return None
    confidence, predicted = torch.max(probabilities, dim=0)
    return int(predicted.item()), float(confidence.item())


def _render_classification_summary(axis, tensor, title: str, reference_label: int | None = None) -> bool:
    """Render one classification target as a compact text summary."""

    label_text = _classification_label_text(tensor)
    prediction = _classification_prediction_summary(tensor)
    if prediction is None and label_text is None:
        return False
    lines = []
    if prediction is not None:
        predicted_label, confidence = prediction
        lines.append(f'pred class: {predicted_label}')
        lines.append(f'confidence: {confidence:.4f}')
    elif label_text is not None:
        lines.append(f'class: {label_text}')
    if reference_label is not None:
        lines.append(f'ref: {reference_label}')
    axis.text(0.5, 0.5, '\n'.join(lines), ha='center', va='center')
    axis.set_title(title)
    axis.set_axis_off()
    return True


def _render_classification_probabilities(axis, tensor, title: str, reference_label: int | None = None) -> bool:
    """Render one classification prediction as a probability bar chart."""

    probabilities = _classification_probabilities(tensor)
    prediction = _classification_prediction_summary(tensor)
    if probabilities is None or probabilities.numel() == 0 or probabilities.numel() > 32:
        return False
    values = probabilities.tolist()
    colors = ['tab:blue'] * len(values)
    if prediction is not None:
        predicted_label, _confidence = prediction
        if 0 <= predicted_label < len(colors):
            colors[predicted_label] = 'tab:green'
    if reference_label is not None and 0 <= reference_label < len(colors):
        colors[reference_label] = 'tab:orange'
    axis.bar(list(range(len(values))), values, color=colors, alpha=0.85)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel('class')
    axis.set_ylabel('probability')
    axis.set_title(title)
    axis.grid(True, axis='y', alpha=0.25)
    notes = []
    if prediction is not None:
        predicted_label, confidence = prediction
        notes.append(f'pred={predicted_label} ({confidence:.3f})')
    if reference_label is not None:
        notes.append(f'ref={reference_label}')
    if notes:
        axis.text(0.98, 0.95, '\n'.join(notes), ha='right', va='top', transform=axis.transAxes)
    return True


def _apply_integer_x_grid(axis) -> None:
    """Snap attack-plot x-axis ticks and grid lines to integer sample indices."""

    try:
        from matplotlib.ticker import MultipleLocator
    except Exception:
        return
    axis.xaxis.set_major_locator(MultipleLocator(1))


def _render_target_panel(
    axis,
    tensor,
    title: str,
    reference_label: int | None = None,
    *,
    classification_mode: str = 'auto',
) -> None:
    """Render one target tensor either as class text, probabilities, summary, or a line plot."""

    label_text = _classification_label_text(tensor)
    probabilities = _classification_probabilities(tensor)
    if probabilities is not None and probabilities.numel() <= 32:
        if classification_mode == 'probabilities' and _render_classification_probabilities(axis, tensor, title, reference_label=reference_label):
            return
        if _render_classification_summary(axis, tensor, title, reference_label=reference_label):
            return
    if label_text is not None:
        axis.text(0.5, 0.5, f'class: {label_text}', ha='center', va='center')
        axis.set_title(title)
        axis.set_axis_off()
        return
    series = _to_series(tensor)
    if series:
        axis.plot(series, linewidth=2.0)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        return
    axis.text(0.5, 0.5, 'target unavailable', ha='center', va='center')
    axis.set_title(title)
    axis.set_axis_off()


def _attack_reconstruction_figure(result):
    """Return a matplotlib figure for attack reconstruction diagnostics."""

    real_x = getattr(result, "plot_reference_x", None)
    if real_x is None:
        real_x = getattr(result, "reference_x", None)
    reconstructed_x = getattr(result, "plot_reconstructed_x", getattr(result, "reconstructed_x", None))
    if real_x is None or reconstructed_x is None:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        logger.warning("Attack plot disabled: {}", exc)
        return None

    real_y = getattr(result, "plot_reference_y", None)
    if real_y is None:
        real_y = getattr(result, "reference_y", None)
    reconstructed_y = getattr(result, "plot_reconstructed_y", getattr(result, "reconstructed_y", None))
    reference_label = getattr(result, "reference_label", None) or "reference"
    title = (
        f"{getattr(result, 'name', 'attack')} client={getattr(result, 'client_id', 'na')} "
        f"round={getattr(result, 'round_index', 'na')} sample={getattr(result, 'sample_index', 'na')}"
    )

    if _image_tensor(real_x) is not None and _image_tensor(reconstructed_x) is not None:
        figure, axes = plt.subplots(2, 2, figsize=(8, 6), height_ratios=[3, 2])
        _render_image(axes[0, 0], real_x, f"{reference_label}_x")
        _render_image(axes[0, 1], reconstructed_x, 'reconstructed_x')
        reference_class = None
        real_label_text = _classification_label_text(real_y)
        if real_label_text is not None and ',' not in real_label_text:
            try:
                reference_class = int(real_label_text)
            except ValueError:
                reference_class = None
        _render_target_panel(axes[1, 0], real_y, f"{reference_label}_y")
        _render_target_panel(
            axes[1, 1],
            reconstructed_y,
            'reconstructed_y',
            reference_label=reference_class,
            classification_mode='probabilities',
        )
    else:
        figure, axes = plt.subplots(1, 2, figsize=(12, 3))
        axes[0].plot(_to_series(real_x), label=f"{reference_label}_x", linewidth=2.0)
        axes[0].plot(_to_series(reconstructed_x), label='reconstructed_x', linewidth=2.0)
        axes[0].set_title('Input reconstruction')
        _apply_integer_x_grid(axes[0])
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')
        y_real_series = _to_series(real_y)
        y_recon_series = _to_series(reconstructed_y)
        if y_real_series:
            axes[1].plot(y_real_series, label=f"{reference_label}_y", linewidth=2.0)
        if y_recon_series:
            axes[1].plot(y_recon_series, label='reconstructed_y', linewidth=2.0)
        axes[1].set_title('Target reconstruction')
        _apply_integer_x_grid(axes[1])
        axes[1].grid(True, alpha=0.3)
        if y_real_series or y_recon_series:
            axes[1].legend(loc='best')
    figure.suptitle(title)
    figure.tight_layout()
    return figure
