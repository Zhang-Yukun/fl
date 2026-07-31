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


class PatchTSTForecaster(nn.Module):
    """Compact PatchTST-style forecaster for univariate or multivariate windows.

    The implementation follows the core PatchTST idea: split each channel into
    temporal patches, embed patches, run a Transformer encoder, and project the
    flattened patch representations to the prediction horizon.

    Example:
        ``PatchTSTForecaster(21, 7, 1)(torch.zeros(4, 21, 1))`` returns
        a tensor shaped ``[4, 7, 1]``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        patch_len: int = 7,
        stride: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        e_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.1,
    ):
        """Create a PatchTST-style encoder and forecasting head."""

        super().__init__()
        if patch_len > seq_len:
            raise ValueError("patch_len must be <= seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.patch_len = patch_len
        self.stride = stride
        self.patch_count = ((seq_len - patch_len) // stride) + 1
        self.patch_embedding = nn.Linear(patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.patch_count * d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict future windows from ``[batch, seq_len, channels]`` input."""

        means = x.mean(dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - means) / stdev
        batch, _, channels = x_norm.shape
        patches = x_norm.permute(0, 2, 1).unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(batch * channels, self.patch_count, self.patch_len)
        encoded = self.encoder(self.patch_embedding(patches))
        pred = self.head(encoded).reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        return pred * stdev[:, :1, :] + means[:, :1, :]


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
    if name == "patchtst":
        return PatchTSTForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=channels,
            patch_len=int(model_cfg.get("patch_len", 7)),
            stride=int(model_cfg.get("stride", 4)),
            d_model=int(model_cfg.get("d_model", 32)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            e_layers=int(model_cfg.get("e_layers", 1)),
            d_ff=int(model_cfg.get("d_ff", 64)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown model name: {name}")

