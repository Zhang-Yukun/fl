"""Image classification dataset loading from pre-split client tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class TensorClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Classification dataset backed by saved image and label tensors."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have matching leading dimension")
        self.images = images.to(torch.float32)
        self.labels = labels.to(torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class _ConcatLoader:
    """Small iterable that presents multiple DataLoaders as one validation/test stream."""

    def __init__(self, loaders: list[DataLoader]):
        self.loaders = loaders
        self.class_names = getattr(loaders[0], 'class_names', None) if loaders else None

    def __iter__(self):
        for loader in self.loaders:
            yield from loader

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)


def _seed_worker(worker_id: int) -> None:
    import random
    import numpy as np

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_kwargs(batch_size: int, shuffle: bool, num_workers: int, seed: int | None = None, identity: str = '') -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
    }
    if seed is None:
        return kwargs
    offset = sum((index + 1) * ord(char) for index, char in enumerate(identity))
    generator = torch.Generator()
    generator.manual_seed(int(seed) + offset)
    kwargs['generator'] = generator
    if num_workers > 0:
        kwargs['worker_init_fn'] = _seed_worker
    return kwargs


def read_split_payload(path: str | Path) -> dict[str, torch.Tensor]:
    """Load one saved ``train.pt``/``val.pt``/``test.pt`` payload."""

    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    if not {'images', 'labels'}.issubset(payload):
        raise ValueError(f"Split payload {path} must contain images and labels")
    return payload


def _attach_loader_metadata(loader: DataLoader, class_names: list[str] | None) -> DataLoader:
    setattr(loader, 'class_names', class_names)
    return loader


def build_federated_image_classification_loaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], Any, Any]:
    """Build per-client train loaders plus shared validation/test loaders."""

    data_cfg = config.get('data', {})
    split_dir = Path(data_cfg['split_dir'])
    clients = list(data_cfg.get('clients', ['client1', 'client2', 'client3']))
    batch_size = int(data_cfg.get('batch_size', 64))
    num_workers = int(data_cfg.get('num_workers', 0))
    shuffle_train = bool(data_cfg.get('shuffle_train', True))
    seed = config.get('runtime', {}).get('seed')
    summary_path = split_dir / 'summary.json'
    class_names = None
    if summary_path.exists():
        import json

        class_names = json.loads(summary_path.read_text(encoding='utf-8')).get('class_names')

    train_loaders: dict[str, DataLoader] = {}
    val_loaders: list[DataLoader] = []
    test_loaders: list[DataLoader] = []
    for client_id in clients:
        client_dir = split_dir / 'clients' / client_id
        train_payload = read_split_payload(client_dir / 'train.pt')
        val_payload = read_split_payload(client_dir / 'val.pt')
        test_payload = read_split_payload(client_dir / 'test.pt')
        train_loader = _attach_loader_metadata(
            DataLoader(TensorClassificationDataset(train_payload['images'], train_payload['labels']), **_loader_kwargs(batch_size, shuffle_train, num_workers, seed, client_id + ':train')),
            class_names,
        )
        val_loader = _attach_loader_metadata(
            DataLoader(TensorClassificationDataset(val_payload['images'], val_payload['labels']), **_loader_kwargs(batch_size, False, num_workers, seed, client_id + ':val')),
            class_names,
        )
        test_loader = _attach_loader_metadata(
            DataLoader(TensorClassificationDataset(test_payload['images'], test_payload['labels']), **_loader_kwargs(batch_size, False, num_workers, seed, client_id + ':test')),
            class_names,
        )
        train_loaders[client_id] = train_loader
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
    return train_loaders, _ConcatLoader(val_loaders), _ConcatLoader(test_loaders)
