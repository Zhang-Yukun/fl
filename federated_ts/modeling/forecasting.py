"""Model registry for time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from federated_ts.utils.peft import is_fedpetuning


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


class ReferencePatchTSTForecaster(nn.Module):
    """Wrapper around the vendored full PatchTST reference implementation.

    The implementation is copied into
    ``federated_ts.modeling.reference_patchtst`` from ``Time-Series-Prediction``
    so training does not import across project directories.

    Example:
        ``ReferencePatchTSTForecaster(config)(torch.zeros(4, 21, 1))`` returns
        a tensor shaped ``[4, 7, 1]``.
    """

    def __init__(self, config: dict[str, Any]):
        """Create the vendored reference PatchTST model from framework config."""

        super().__init__()
        from types import SimpleNamespace

        from federated_ts.modeling.reference_patchtst.patchtst import Model as PatchTSTModel

        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        args = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=int(data_cfg.get("seq_len", 21)),
            pred_len=int(data_cfg.get("pred_len", 7)),
            d_model=int(model_cfg.get("d_model", 384)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            factor=int(model_cfg.get("factor", 3)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            d_ff=int(model_cfg.get("d_ff", 2048)),
            e_layers=int(model_cfg.get("e_layers", 2)),
            activation=str(model_cfg.get("activation", "gelu")),
        )
        self.model = PatchTSTModel(
            args,
            patch_len=int(model_cfg.get("patch_len", 16)),
            stride=int(model_cfg.get("stride", 8)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward input windows through the vendored PatchTST model."""

        return self.model(x)


class BottleneckAdapter(nn.Module):
    """Small bottleneck adapter used for FedPETuning-style communication reduction."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int, dropout: float = 0.1):
        """Create a residual bottleneck adapter on the hidden dimension."""

        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual adapter to the last tensor dimension."""

        residual = x
        x = self.down(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + x


class FedPETuningPatchTSTForecaster(nn.Module):
    """FedPETuning-style PatchTST with frozen backbone and small trainable modules."""

    def __init__(self, config: dict[str, Any]):
        """Create the frozen PatchTST backbone and trainable adapter modules."""

        super().__init__()
        from types import SimpleNamespace

        from federated_ts.modeling.reference_patchtst.patchtst import Model as PatchTSTModel

        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        peft_cfg = model_cfg.get("peft", {})
        args = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=int(data_cfg.get("seq_len", 21)),
            pred_len=int(data_cfg.get("pred_len", 7)),
            d_model=int(model_cfg.get("d_model", 384)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            factor=int(model_cfg.get("factor", 3)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            d_ff=int(model_cfg.get("d_ff", 2048)),
            e_layers=int(model_cfg.get("e_layers", 2)),
            activation=str(model_cfg.get("activation", "gelu")),
        )
        self.backbone = PatchTSTModel(
            args,
            patch_len=int(model_cfg.get("patch_len", 16)),
            stride=int(model_cfg.get("stride", 8)),
        )
        self.pred_len = int(data_cfg.get("pred_len", 7))
        self.adapter = BottleneckAdapter(
            hidden_dim=int(model_cfg.get("d_model", 384)),
            bottleneck_dim=int(peft_cfg.get("bottleneck_dim", 32)),
            dropout=float(peft_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        )
        train_head = bool(peft_cfg.get("train_head", True))
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        if train_head:
            for param in self.backbone.head.parameters():
                param.requires_grad_(True)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Forward input windows through frozen PatchTST plus trainable adapters."""

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.backbone.patch_embedding(x_enc)
        enc_out, _ = self.backbone.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = self.adapter(enc_out)
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.backbone.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:, -self.pred_len :, :]


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
        if is_fedpetuning(config):
            return FedPETuningPatchTSTForecaster(config)
        return ReferencePatchTSTForecaster(config)
    raise ValueError(f"Unknown model name: {name}")
