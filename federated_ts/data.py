"""Rare-earth time-series data loading for three-client federation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


OXIDE_COLUMNS = {
    "Nd2O3": "氧化钕",
    "CeO2": "氧化铈",
    "La2O3": "氧化镧",
}


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        std = values.std(axis=0)
        std[std == 0] = 1.0
        return cls(values.mean(axis=0), std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset returning encoder and prediction windows."""

    def __init__(self, values: np.ndarray, seq_len: int, pred_len: int):
        if values.ndim == 1:
            values = values[:, None]
        if len(values) < seq_len + pred_len:
            raise ValueError("Not enough observations for requested seq_len + pred_len")
        self.values = values.astype("float32")
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.values[index : index + self.seq_len]
        y = self.values[index + self.seq_len : index + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def read_price_frame(csv_path: str | Path) -> pd.DataFrame:
    """Read a price CSV and normalize dates/ordering."""

    frame = pd.read_csv(csv_path)
    if "date" not in frame.columns:
        raise ValueError("CSV must contain a 'date' column")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame = frame.ffill().bfill()
    return frame


def split_array(values: np.ndarray, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(len(values) * train_ratio)
    val_end = train_end + int(len(values) * val_ratio)
    return values[:train_end], values[train_end:val_end], values[val_end:]


def make_loaders(
    values: np.ndarray,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, Standardizer]:
    train_raw, val_raw, test_raw = split_array(values, train_ratio, val_ratio)
    scaler = Standardizer.fit(train_raw)
    train = scaler.transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)
    return (
        DataLoader(WindowDataset(train, seq_len, pred_len), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(WindowDataset(val, seq_len, pred_len), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(WindowDataset(test, seq_len, pred_len), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        scaler,
    )


def build_federated_loaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], DataLoader, DataLoader]:
    """Build one train loader per oxide client and shared server validation/test loaders."""

    data_cfg = config["data"]
    frame = read_price_frame(data_cfg["csv_path"])
    clients = data_cfg.get("clients", list(OXIDE_COLUMNS.keys()))
    seq_len = int(data_cfg.get("seq_len", 21))
    pred_len = int(data_cfg.get("pred_len", 7))
    batch_size = int(data_cfg.get("batch_size", 32))
    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    val_ratio = float(data_cfg.get("val_ratio", 0.15))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loaders: dict[str, DataLoader] = {}
    val_loaders = []
    test_loaders = []
    for client in clients:
        column = OXIDE_COLUMNS.get(client, client)
        train, val, test, _ = make_loaders(
            frame[[column]].to_numpy(dtype="float32"),
            seq_len,
            pred_len,
            batch_size,
            train_ratio,
            val_ratio,
            num_workers,
        )
        train_loaders[client] = train
        val_loaders.append(val)
        test_loaders.append(test)
    return train_loaders, _ConcatLoader(val_loaders), _ConcatLoader(test_loaders)


class _ConcatLoader:
    """Small iterable that presents multiple DataLoaders as one validation/test stream."""

    def __init__(self, loaders: list[DataLoader]):
        self.loaders = loaders

    def __iter__(self):
        for loader in self.loaders:
            yield from loader

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)

