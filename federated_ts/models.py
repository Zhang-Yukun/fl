"""Model registry for time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


@dataclass
class ModelSpec:
    """Registry item pairing a model name with a builder.

    Example:
        ``ModelSpec("mlp", lambda cfg: ForecastMLP(4, 2, 1))``.
    """

    name: str
    builder: Callable[[dict[str, Any]], nn.Module]


class ForecastMLP(nn.Module):
    """A compact baseline forecaster mapping ``[batch, seq_len, channels]`` to future values."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, hidden_size: int = 64, dropout: float = 0.1):
        """Create the MLP baseline with fixed input/output window sizes."""

        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len * channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future windows from input windows."""

        return self.net(x).reshape(x.shape[0], self.pred_len, self.channels)


class LSTMForecaster(nn.Module):
    """LSTM forecaster with an autoregressive-size projection head."""

    def __init__(self, seq_len: int, pred_len: int, channels: int, hidden_size: int = 64, num_layers: int = 1):
        """Create the LSTM forecaster with a projection head."""

        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, pred_len * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future windows using the final LSTM hidden state."""

        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).reshape(x.shape[0], self.pred_len, self.channels)


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a model from ``config['model']``."""

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    seq_len = int(data_cfg.get("seq_len", 21))
    pred_len = int(data_cfg.get("pred_len", 7))
    channels = int(model_cfg.get("channels", 1))
    name = str(model_cfg.get("name", "mlp")).lower()
    if name == "mlp":
        return ForecastMLP(seq_len, pred_len, channels, int(model_cfg.get("hidden_size", 64)), float(model_cfg.get("dropout", 0.1)))
    if name == "lstm":
        return LSTMForecaster(seq_len, pred_len, channels, int(model_cfg.get("hidden_size", 64)), int(model_cfg.get("num_layers", 1)))
    raise ValueError(f"Unknown model name: {name}")

