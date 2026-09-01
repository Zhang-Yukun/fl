"""Image classification dataset loading from pre-split client tensors."""

from __future__ import annotations

import json
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


def _read_split_summary(split_dir: str | Path) -> dict[str, Any]:
    """Read the optional split summary for one prepared image dataset."""

    summary_path = Path(split_dir) / 'summary.json'
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding='utf-8'))


def _build_loader_from_payload(
    payload: dict[str, torch.Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int | None,
    identity: str,
    class_names: list[str] | None,
) -> DataLoader:
    """Build one classification DataLoader from a saved tensor payload."""

    return _attach_loader_metadata(
        DataLoader(
            TensorClassificationDataset(payload['images'], payload['labels']),
            **_loader_kwargs(batch_size, shuffle, num_workers, seed, identity),
        ),
        class_names,
    )


def _load_server_or_merged_eval_payloads(split_dir: Path, clients: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Load shared validation/test payloads without touching any client train split."""

    server_dir = split_dir / 'server'
    val_payload = read_split_payload(server_dir / 'val.pt') if (server_dir / 'val.pt').exists() else None
    test_payload = read_split_payload(server_dir / 'test.pt') if (server_dir / 'test.pt').exists() else None
    if val_payload is not None:
        return val_payload, test_payload

    val_images = []
    val_labels = []
    test_images = []
    test_labels = []
    have_test_payloads = True
    for client_id in clients:
        client_dir = split_dir / 'clients' / client_id
        client_val = read_split_payload(client_dir / 'val.pt')
        val_images.append(client_val['images'])
        val_labels.append(client_val['labels'])
        client_test_path = client_dir / 'test.pt'
        if client_test_path.exists():
            client_test = read_split_payload(client_test_path)
            test_images.append(client_test['images'])
            test_labels.append(client_test['labels'])
        else:
            have_test_payloads = False
    merged_test_payload = None
    if have_test_payloads and test_images:
        merged_test_payload = {'images': torch.cat(test_images, dim=0), 'labels': torch.cat(test_labels, dim=0)}
    return (
        {'images': torch.cat(val_images, dim=0), 'labels': torch.cat(val_labels, dim=0)},
        merged_test_payload,
    )


def build_server_image_classification_evaluation_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader | None]:
    """Build only the server-side validation/test loaders for classification."""

    data_cfg = config.get('data', {})
    split_dir = Path(data_cfg['split_dir'])
    clients = list(data_cfg.get('clients', ['client1', 'client2', 'client3']))
    batch_size = int(data_cfg.get('batch_size', 64))
    num_workers = int(data_cfg.get('num_workers', 0))
    seed = config.get('runtime', {}).get('seed')
    summary = _read_split_summary(split_dir)
    class_names = summary.get('class_names')
    val_payload, test_payload = _load_server_or_merged_eval_payloads(split_dir, clients)
    val_loader = _build_loader_from_payload(
        val_payload,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        identity='server:val',
        class_names=class_names,
    )
    test_loader = None
    if test_payload is not None:
        test_loader = _build_loader_from_payload(
            test_payload,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            identity='server:test',
            class_names=class_names,
        )
    return val_loader, test_loader


def build_federated_image_classification_loaders(config: dict[str, Any]) -> tuple[dict[str, DataLoader], Any, Any]:
    """Build per-client train loaders plus shared validation/test loaders."""

    data_cfg = config.get('data', {})
    split_dir = Path(data_cfg['split_dir'])
    clients = list(data_cfg.get('clients', ['client1', 'client2', 'client3']))
    batch_size = int(data_cfg.get('batch_size', 64))
    num_workers = int(data_cfg.get('num_workers', 0))
    shuffle_train = bool(data_cfg.get('shuffle_train', True))
    seed = config.get('runtime', {}).get('seed')
    summary = _read_split_summary(split_dir)
    class_names = summary.get('class_names')

    train_loaders: dict[str, DataLoader] = {}
    for client_id in clients:
        client_dir = split_dir / 'clients' / client_id
        train_payload = read_split_payload(client_dir / 'train.pt')
        train_loaders[client_id] = _build_loader_from_payload(
            train_payload,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            seed=seed,
            identity=client_id + ':train',
            class_names=class_names,
        )

    val_loader, test_loader = build_server_image_classification_evaluation_loaders(config)
    return train_loaders, val_loader, test_loader


def build_client_image_classification_train_loader(config: dict[str, Any], client_id: str) -> DataLoader:
    """Build only one client's local training loader from a prepared split directory."""

    data_cfg = config.get('data', {})
    split_dir = Path(data_cfg['split_dir'])
    batch_size = int(data_cfg.get('batch_size', 64))
    num_workers = int(data_cfg.get('num_workers', 0))
    shuffle_train = bool(data_cfg.get('shuffle_train', True))
    seed = config.get('runtime', {}).get('seed')
    summary = _read_split_summary(split_dir)
    class_names = summary.get('class_names')
    client_dir = split_dir / 'clients' / client_id
    if not client_dir.exists():
        raise ValueError(f'Unknown client_id {client_id}; expected local split under {client_dir}')
    train_payload = read_split_payload(client_dir / 'train.pt')
    return _build_loader_from_payload(
        train_payload,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        seed=seed,
        identity=client_id + ':train',
        class_names=class_names,
    )


def summarize_image_classification_training(config: dict[str, Any]) -> dict[str, int]:
    """Return total client count and total training samples for one prepared split."""

    data_cfg = config.get('data', {})
    split_dir = Path(data_cfg['split_dir'])
    clients = list(data_cfg.get('clients', ['client1', 'client2', 'client3']))
    summary = _read_split_summary(split_dir)
    client_summaries = summary.get('clients')
    if isinstance(client_summaries, dict) and client_summaries:
        total_clients = int(summary.get('num_clients') or len(client_summaries))
        total_train_samples = 0
        for client_id in clients:
            payload = client_summaries.get(client_id) or {}
            train_payload = payload.get('train') or {}
            total_train_samples += int(train_payload.get('rows', 0) or 0)
        if total_train_samples > 0:
            return {
                'total_clients': total_clients,
                'total_train_samples': total_train_samples,
            }
    train_loaders, _, _ = build_federated_image_classification_loaders(config)
    return {
        'total_clients': len(train_loaders),
        'total_train_samples': sum(len(loader.dataset) for loader in train_loaders.values()),
    }
