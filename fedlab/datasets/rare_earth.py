"""Rare-earth time-series data loading for three-client federation."""

from __future__ import annotations

import json
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
    """Feature-wise standardization statistics.

    Example:
        ``scaler = Standardizer.fit(values); scaled = scaler.transform(values)``.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        """Estimate mean and standard deviation from training values."""

        std = values.std(axis=0)
        std[std == 0] = 1.0
        return cls(values.mean(axis=0), std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Scale raw values with stored statistics."""

        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map standardized values back to the original price scale."""

        return values * self.std + self.mean


def inverse_transform_tensor(values: torch.Tensor, scaler: Standardizer | None) -> torch.Tensor:
    """Map one standardized tensor back to the original value scale.

    Example:
        ``restored = inverse_transform_tensor(batch, loader.scaler)``.
    """

    if scaler is None:
        return values.detach().cpu().clone()
    tensor = values.detach().cpu().to(torch.float32)
    mean = torch.as_tensor(scaler.mean, dtype=tensor.dtype).reshape(1, 1, -1)
    std = torch.as_tensor(scaler.std, dtype=tensor.dtype).reshape(1, 1, -1)
    while mean.ndim < tensor.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return tensor * std + mean


def attach_loader_scaler(loader: DataLoader, scaler: Standardizer) -> DataLoader:
    """Attach one fitted scaler onto a loader for visualization recovery."""

    setattr(loader, "scaler", scaler)
    return loader


def _stable_text_offset(text: str) -> int:
    """Return a deterministic integer offset for a loader identity string."""

    value = 0
    for index, char in enumerate(text):
        value += (index + 1) * ord(char)
    return value


def _seed_worker(worker_id: int) -> None:
    """Propagate torch worker seeds into Python and NumPy."""

    import random

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_kwargs(
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None = None,
    identity: str = '',
) -> dict[str, Any]:
    """Build deterministic DataLoader keyword arguments when a seed is available."""

    kwargs: dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
    }
    if seed is None:
        return kwargs
    generator = torch.Generator()
    generator.manual_seed(int(seed) + _stable_text_offset(identity))
    kwargs['generator'] = generator
    if num_workers > 0:
        kwargs['worker_init_fn'] = _seed_worker
    return kwargs


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset returning encoder and prediction windows."""

    def __init__(self, values: np.ndarray, seq_len: int, pred_len: int):
        """Create sliding windows from a normalized time-series array."""

        if values.ndim == 1:
            values = values[:, None]
        if len(values) < seq_len + pred_len:
            raise ValueError("Not enough observations for requested seq_len + pred_len")
        self.values = values.astype("float32")
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        """Return the number of valid sliding windows."""

        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input and prediction target window pair."""

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
    """Split an array into train, validation, and test partitions.

    Example:
        ``train, val, test = split_array(values, 0.7, 0.15)``.
    """

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
    shuffle_train: bool = True,
    seed: int | None = None,
    identity: str = 'series',
) -> tuple[DataLoader, DataLoader, DataLoader, Standardizer]:
    """Build train/validation/test DataLoaders from one time-series array.

    Example:
        ``train, val, test, scaler = make_loaders(values, 21, 7, 32, 0.7, 0.15)``.
    """

    train_raw, val_raw, test_raw = split_array(values, train_ratio, val_ratio)
    scaler = Standardizer.fit(train_raw)
    train = scaler.transform(train_raw)
    val = scaler.transform(val_raw)
    test = scaler.transform(test_raw)
    return (
        attach_loader_scaler(DataLoader(WindowDataset(train, seq_len, pred_len), **_loader_kwargs(batch_size, shuffle_train, num_workers, seed, identity + ':train')), scaler),
        attach_loader_scaler(DataLoader(WindowDataset(val, seq_len, pred_len), **_loader_kwargs(batch_size, False, num_workers, seed, identity + ':val')), scaler),
        attach_loader_scaler(DataLoader(WindowDataset(test, seq_len, pred_len), **_loader_kwargs(batch_size, False, num_workers, seed, identity + ':test')), scaler),
        scaler,
    )


def read_value_frame(csv_path: str | Path) -> pd.DataFrame:
    """Read a single-client CSV with ``date`` and ``value`` columns.

    Example:
        ``frame = read_value_frame("clients/Nd2O3/train.csv")``.
    """

    frame = pd.read_csv(csv_path)
    if not {"date", "value"}.issubset(frame.columns):
        raise ValueError("CSV must contain 'date' and 'value' columns")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.sort_values("date").ffill().bfill().dropna(subset=["value"]).reset_index(drop=True)


def make_loaders_from_splits(
    train_values: np.ndarray,
    val_values: np.ndarray,
    test_values: np.ndarray,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
    seed: int | None = None,
    identity: str = 'series',
) -> tuple[DataLoader, DataLoader, DataLoader, Standardizer]:
    """Build DataLoaders from pre-split train/validation/test arrays.

    Example:
        ``make_loaders_from_splits(train, val, test, 21, 7, 64)`` uses only
        ``train`` to fit scaling statistics.
    """

    scaler = Standardizer.fit(train_values)
    train = scaler.transform(train_values)
    val = scaler.transform(val_values)
    test = scaler.transform(test_values)
    return (
        attach_loader_scaler(DataLoader(WindowDataset(train, seq_len, pred_len), **_loader_kwargs(batch_size, shuffle_train, num_workers, seed, identity + ':train')), scaler),
        attach_loader_scaler(DataLoader(WindowDataset(val, seq_len, pred_len), **_loader_kwargs(batch_size, False, num_workers, seed, identity + ':val')), scaler),
        attach_loader_scaler(DataLoader(WindowDataset(test, seq_len, pred_len), **_loader_kwargs(batch_size, False, num_workers, seed, identity + ':test')), scaler),
        scaler,
    )


def _build_eval_loader_from_values(
    values: np.ndarray,
    scaler: Standardizer,
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    num_workers: int,
    seed: int | None,
    identity: str,
) -> DataLoader:
    """Build one evaluation-only loader using externally provided scaling statistics."""

    normalized = scaler.transform(values)
    return attach_loader_scaler(
        DataLoader(
            WindowDataset(normalized, seq_len, pred_len),
            **_loader_kwargs(batch_size, False, num_workers, seed, identity),
        ),
        scaler,
    )


def _standardizer_from_registration(
    registration_metadata: dict[str, dict[str, Any]] | None,
    client_id: str,
) -> Standardizer:
    """Rebuild a client scaler from the metadata uploaded during registration."""

    metadata = (registration_metadata or {}).get(client_id) or {}
    mean = metadata.get('scale_mean')
    std = metadata.get('scale_std')
    if mean is None or std is None:
        raise ValueError(f'Missing uploaded scaler statistics for client {client_id}')
    return Standardizer(
        mean=np.asarray(mean, dtype='float32').reshape(-1),
        std=np.asarray(std, dtype='float32').reshape(-1),
    )


def _registration_has_scalers(
    registration_metadata: dict[str, dict[str, Any]] | None,
    clients: list[str],
) -> bool:
    """Return whether every expected client uploaded reusable scaling statistics."""

    if not registration_metadata:
        return False
    for client_id in clients:
        metadata = registration_metadata.get(client_id) or {}
        if metadata.get('scale_mean') is None or metadata.get('scale_std') is None:
            return False
    return True


def _split_dir_has_client_train_splits(split_dir: Path, clients: list[str]) -> bool:
    """Return whether every expected client train split is present on disk."""

    return all((split_dir / 'clients' / client_id / 'train.csv').exists() for client_id in clients)


def build_federated_loaders_from_split_dir(data_cfg: dict[str, Any], seed: int | None = None) -> tuple[dict[str, DataLoader], DataLoader, DataLoader]:
    """Build loaders from ``split_dir/clients/<client>/{train,val,test}.csv``.

    Example:
        Configure ``data.split_dir: ../data/rare_earth_rawdata2`` to use
        preprocessed chronological 8:1:1 splits.
    """

    split_dir = Path(data_cfg["split_dir"])
    clients = data_cfg.get("clients", list(OXIDE_COLUMNS.keys()))
    seq_len = int(data_cfg.get("seq_len", 21))
    pred_len = int(data_cfg.get("pred_len", 7))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    shuffle_train = bool(data_cfg.get("shuffle_train", True))
    train_loaders: dict[str, DataLoader] = {}
    val_loaders = []
    test_loaders = []
    for client in clients:
        client_dir = split_dir / "clients" / client
        train = read_value_frame(client_dir / "train.csv")[["value"]].to_numpy(dtype="float32")
        val = read_value_frame(client_dir / "val.csv")[["value"]].to_numpy(dtype="float32")
        test = read_value_frame(client_dir / "test.csv")[["value"]].to_numpy(dtype="float32")
        train_loader, val_loader, test_loader, _ = make_loaders_from_splits(
            train, val, test, seq_len, pred_len, batch_size, num_workers, shuffle_train, seed, client
        )
        train_loaders[client] = train_loader
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
    return train_loaders, _ConcatLoader(val_loaders), _ConcatLoader(test_loaders)


def build_server_rare_earth_evaluation_loaders(
    config: dict[str, Any],
    registration_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[DataLoader | None, DataLoader | None]:
    """Build only the server-side validation/test loaders for forecasting."""

    data_cfg = config['data']
    seed = config.get('runtime', {}).get('seed')
    if 'split_dir' not in data_cfg:
        _, val_loader, test_loader = build_federated_loaders(config)
        return val_loader, test_loader
    split_dir = Path(data_cfg['split_dir'])
    clients = list(data_cfg.get('clients', list(OXIDE_COLUMNS.keys())))
    if not _registration_has_scalers(registration_metadata, clients):
        if not _split_dir_has_client_train_splits(split_dir, clients):
            return None, None
        _, val_loader, test_loader = build_federated_loaders(config)
        return val_loader, test_loader
    seq_len = int(data_cfg.get('seq_len', 21))
    pred_len = int(data_cfg.get('pred_len', 7))
    batch_size = int(data_cfg.get('batch_size', 32))
    num_workers = int(data_cfg.get('num_workers', 0))
    val_loaders = []
    test_loaders = []
    for client_id in clients:
        client_dir = split_dir / 'clients' / client_id
        scaler = _standardizer_from_registration(registration_metadata, client_id)
        val_values = read_value_frame(client_dir / 'val.csv')[['value']].to_numpy(dtype='float32')
        test_values = read_value_frame(client_dir / 'test.csv')[['value']].to_numpy(dtype='float32')
        val_loaders.append(
            _build_eval_loader_from_values(
                val_values,
                scaler,
                seq_len=seq_len,
                pred_len=pred_len,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed,
                identity=client_id + ':server:val',
            )
        )
        test_loaders.append(
            _build_eval_loader_from_values(
                test_values,
                scaler,
                seq_len=seq_len,
                pred_len=pred_len,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=seed,
                identity=client_id + ':server:test',
            )
        )
    return _ConcatLoader(val_loaders), _ConcatLoader(test_loaders)


def build_federated_loaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], DataLoader, DataLoader]:
    """Build one train loader per client and shared server validation/test loaders."""

    data_cfg = config["data"]
    seed = config.get("runtime", {}).get("seed")
    if "split_dir" in data_cfg:
        return build_federated_loaders_from_split_dir(data_cfg, seed=seed)
    frame = read_price_frame(data_cfg["csv_path"])
    clients = data_cfg.get("clients", list(OXIDE_COLUMNS.keys()))
    seq_len = int(data_cfg.get("seq_len", 21))
    pred_len = int(data_cfg.get("pred_len", 7))
    batch_size = int(data_cfg.get("batch_size", 32))
    train_ratio = float(data_cfg.get("train_ratio", 0.7))
    val_ratio = float(data_cfg.get("val_ratio", 0.15))
    num_workers = int(data_cfg.get("num_workers", 0))
    shuffle_train = bool(data_cfg.get("shuffle_train", True))
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
            shuffle_train,
            seed,
            client,
        )
        train_loaders[client] = train
        val_loaders.append(val)
        test_loaders.append(test)
    return train_loaders, _ConcatLoader(val_loaders), _ConcatLoader(test_loaders)


class _ConcatLoader:
    """Small iterable that presents multiple DataLoaders as one validation/test stream."""

    def __init__(self, loaders: list[DataLoader]):
        """Store loaders that should be iterated as one stream."""

        self.loaders = loaders
        self.scaler = getattr(loaders[0], "scaler", None) if loaders else None

    def __iter__(self):
        """Yield batches from each wrapped loader in order."""

        for loader in self.loaders:
            yield from loader

    def __len__(self) -> int:
        """Return the total number of batches across wrapped loaders."""

        return sum(len(loader) for loader in self.loaders)


def build_client_rare_earth_train_loader(config: dict[str, Any], client_id: str) -> DataLoader:
    """Build only one client's local training loader for a prepared split directory."""

    data_cfg = config['data']
    if 'split_dir' not in data_cfg:
        train_loaders, _, _ = build_federated_loaders(config)
        if client_id not in train_loaders:
            raise ValueError(f'Unknown client_id {client_id}; expected one of {sorted(train_loaders)}')
        return train_loaders[client_id]
    split_dir = Path(data_cfg['split_dir'])
    seq_len = int(data_cfg.get('seq_len', 21))
    pred_len = int(data_cfg.get('pred_len', 7))
    batch_size = int(data_cfg.get('batch_size', 32))
    num_workers = int(data_cfg.get('num_workers', 0))
    shuffle_train = bool(data_cfg.get('shuffle_train', True))
    seed = config.get('runtime', {}).get('seed')
    client_dir = split_dir / 'clients' / client_id
    if not client_dir.exists():
        raise ValueError(f'Unknown client_id {client_id}; expected local split under {client_dir}')
    train = read_value_frame(client_dir / 'train.csv')[['value']].to_numpy(dtype='float32')
    train_loader, _, _, _ = make_loaders_from_splits(
        train,
        train,
        train,
        seq_len,
        pred_len,
        batch_size,
        num_workers,
        shuffle_train,
        seed,
        client_id,
    )
    return train_loader


def summarize_rare_earth_training(config: dict[str, Any]) -> dict[str, int]:
    """Return total client count and total train-window count for one experiment."""

    data_cfg = config['data']
    if 'split_dir' not in data_cfg:
        train_loaders, _, _ = build_federated_loaders(config)
        return {
            'total_clients': len(train_loaders),
            'total_train_samples': sum(_loader_num_samples(loader) for loader in train_loaders.values()),
        }
    split_dir = Path(data_cfg['split_dir'])
    clients = list(data_cfg.get('clients', list(OXIDE_COLUMNS.keys())))
    seq_len = int(data_cfg.get('seq_len', 21))
    pred_len = int(data_cfg.get('pred_len', 7))
    summary_path = split_dir / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        row_summary = summary.get('rows')
        if isinstance(row_summary, dict) and row_summary:
            total_train_samples = 0
            for client_id in clients:
                counts = row_summary.get(client_id) or {}
                train_rows = int(counts.get('train', 0) or 0)
                total_train_samples += max(0, train_rows - seq_len - pred_len + 1)
            if total_train_samples > 0:
                return {
                    'total_clients': len(clients),
                    'total_train_samples': total_train_samples,
                }
    train_loaders, _, _ = build_federated_loaders(config)
    return {
        'total_clients': len(train_loaders),
        'total_train_samples': sum(_loader_num_samples(loader) for loader in train_loaders.values()),
    }


def _loader_num_samples(loader: Any) -> int:
    """Return the number of samples carried by one loader-like object."""

    dataset = getattr(loader, 'dataset', None)
    return len(dataset) if dataset is not None else len(loader)
