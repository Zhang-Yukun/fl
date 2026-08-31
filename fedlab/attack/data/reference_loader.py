"""Load full client-local reference sets for offline replay attacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass
class ClientReferenceSet:
    """Full client-local reference set used for offline replay scoring."""

    inputs: torch.Tensor
    targets: torch.Tensor | None
    scale_mean: list[float] | None = None
    scale_std: list[float] | None = None


class _WindowDataset:
    def __init__(self, values: np.ndarray, seq_len: int, pred_len: int):
        if values.ndim == 1:
            values = values[:, None]
        if len(values) < seq_len + pred_len:
            raise ValueError("Not enough observations for requested seq_len + pred_len")
        self.values = values.astype("float32")
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

    def __len__(self) -> int:
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.values[index : index + self.seq_len]
        y = self.values[index + self.seq_len : index + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


class _Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, values: np.ndarray) -> "_Standardizer":
        std = values.std(axis=0)
        std[std == 0] = 1.0
        return cls(values.mean(axis=0), std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


_OXIDE_COLUMNS = {
    "Nd2O3": "氧化钕",
    "CeO2": "氧化铈",
    "La2O3": "氧化镧",
}


def _read_value_frame(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if not {"date", "value"}.issubset(frame.columns):
        raise ValueError("CSV must contain 'date' and 'value' columns")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.sort_values("date").ffill().bfill().dropna(subset=["value"]).reset_index(drop=True)


def _read_price_frame(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "date" not in frame.columns:
        raise ValueError("CSV must contain a 'date' column")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame.ffill().bfill()


def _split_array(values: np.ndarray, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(len(values) * train_ratio)
    val_end = train_end + int(len(values) * val_ratio)
    return values[:train_end], values[train_end:val_end], values[val_end:]


def _stack_window_dataset(dataset: _WindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for index in range(len(dataset)):
        x, y = dataset[index]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def _load_forecasting_reference_set(config: dict[str, Any], client_id: str) -> ClientReferenceSet:
    data_cfg = config.get("data", {})
    seq_len = int(data_cfg.get("seq_len", 21))
    pred_len = int(data_cfg.get("pred_len", 7))
    if "split_dir" in data_cfg:
        split_dir = Path(data_cfg["split_dir"])
        client_dir = split_dir / "clients" / client_id
        train_values = _read_value_frame(client_dir / "train.csv")[["value"]].to_numpy(dtype="float32")
    else:
        frame = _read_price_frame(data_cfg["csv_path"])
        column = _OXIDE_COLUMNS.get(client_id, client_id)
        values = frame[[column]].to_numpy(dtype="float32")
        train_values, _, _ = _split_array(
            values,
            float(data_cfg.get("train_ratio", 0.7)),
            float(data_cfg.get("val_ratio", 0.15)),
        )
    scaler = _Standardizer.fit(train_values)
    dataset = _WindowDataset(scaler.transform(train_values), seq_len, pred_len)
    reference_inputs, reference_targets = _stack_window_dataset(dataset)
    return ClientReferenceSet(
        inputs=reference_inputs,
        targets=reference_targets,
        scale_mean=[float(value) for value in scaler.mean.reshape(-1).tolist()],
        scale_std=[float(value) for value in scaler.std.reshape(-1).tolist()],
    )


def _read_split_payload(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not {"images", "labels"}.issubset(payload):
        raise ValueError(f"Split payload {path} must contain images and labels")
    return payload


def _load_classification_reference_set(config: dict[str, Any], client_id: str) -> ClientReferenceSet:
    data_cfg = config.get("data", {})
    split_dir = Path(data_cfg["split_dir"])
    client_dir = split_dir / "clients" / client_id
    if not client_dir.exists():
        raise ValueError(f"Unknown client_id {client_id}; expected local split under {client_dir}")
    payload = _read_split_payload(client_dir / "train.pt")
    return ClientReferenceSet(
        inputs=payload["images"].to(torch.float32),
        targets=payload["labels"].to(torch.long),
        scale_mean=None,
        scale_std=None,
    )


def load_client_reference_set(config: dict[str, Any], client_id: str) -> ClientReferenceSet:
    """Load one client's full local training reference set for offline replay."""

    task_type = str(config.get("task", {}).get("type", "forecasting")).lower()
    if task_type == "classification":
        return _load_classification_reference_set(config, client_id)
    if task_type == "forecasting":
        return _load_forecasting_reference_set(config, client_id)
    raise ValueError(f"Unsupported task.type for attack replay reference loading: {task_type}")
