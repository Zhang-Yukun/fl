#!/usr/bin/env python3
"""Download and split image classification datasets for three-client federation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torchvision import datasets, transforms


_DATASET_BUILDERS = {
    "mnist": datasets.MNIST,
    "cifar10": datasets.CIFAR10,
}

_DATASET_CLIENT_PREFIX = {
    'mnist': 'm',
    'cifar10': 'c',
}


def reset_output_dir(output_dir: Path) -> None:
    """Remove stale generated client artifacts from a previous preparation run."""

    if output_dir.exists():
        shutil.rmtree(output_dir)


def _partition_indices(num_items: int, num_parts: int, generator: torch.Generator) -> list[torch.Tensor]:
    """Return a deterministic random partition of ``range(num_items)``."""

    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    perm = torch.randperm(num_items, generator=generator)
    sizes = [num_items // num_parts] * num_parts
    for index in range(num_items % num_parts):
        sizes[index] += 1
    parts: list[torch.Tensor] = []
    offset = 0
    for size in sizes:
        parts.append(perm[offset : offset + size])
        offset += size
    return parts


def _default_client_ids(dataset_name: str, num_clients: int) -> list[str]:
    """Return the persisted client ids for one prepared image dataset."""

    dataset_key = str(dataset_name).lower()
    prefix = _DATASET_CLIENT_PREFIX.get(dataset_key)
    if prefix is None:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    return [f'{prefix}{index}' for index in range(1, int(num_clients) + 1)]


def build_client_split_payloads(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    dataset_name: str,
    num_clients: int = 3,
    val_ratio: float = 0.1,
    seed: int = 2026,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Split image tensors into train/val/test shards for each client."""

    if not 0.0 < float(val_ratio) < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_images.shape[0] != train_labels.shape[0]:
        raise ValueError("train_images and train_labels must have matching length")
    if test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("test_images and test_labels must have matching length")

    generator = torch.Generator().manual_seed(int(seed))
    train_parts = _partition_indices(int(train_images.shape[0]), int(num_clients), generator)
    test_parts = _partition_indices(int(test_images.shape[0]), int(num_clients), generator)
    client_ids = _default_client_ids(dataset_name, num_clients)
    payloads: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for client_id, train_indices, test_indices in zip(client_ids, train_parts, test_parts):
        val_count = max(1, int(round(train_indices.numel() * float(val_ratio))))
        if train_indices.numel() - val_count < 1:
            raise ValueError(f"Client {client_id} does not have enough train samples after validation split")
        train_perm = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        val_indices = train_perm[:val_count]
        client_train_indices = train_perm[val_count:]
        payloads[client_id] = {
            "train": {
                "images": train_images.index_select(0, client_train_indices).clone(),
                "labels": train_labels.index_select(0, client_train_indices).clone(),
            },
            "val": {
                "images": train_images.index_select(0, val_indices).clone(),
                "labels": train_labels.index_select(0, val_indices).clone(),
            },
            "test": {
                "images": test_images.index_select(0, test_indices).clone(),
                "labels": test_labels.index_select(0, test_indices).clone(),
            },
        }
    return payloads


def _load_torchvision_dataset(dataset_name: str, raw_root: Path, train: bool):
    """Load one torchvision dataset split as normalized tensors."""

    builder = _DATASET_BUILDERS.get(str(dataset_name).lower())
    if builder is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset = builder(root=str(raw_root), train=train, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    images = []
    labels = []
    for batch_images, batch_labels in loader:
        images.append(batch_images)
        labels.append(batch_labels.to(torch.long))
    return torch.cat(images, dim=0), torch.cat(labels, dim=0), getattr(dataset, "classes", None)


def _combine_server_payload(
    payloads: dict[str, dict[str, dict[str, torch.Tensor]]],
    split_name: str,
) -> dict[str, torch.Tensor]:
    """Combine one split across all clients into the shared server view."""

    client_ids = list(payloads.keys())
    return {
        'images': torch.cat([payloads[client_id][split_name]['images'] for client_id in client_ids], dim=0),
        'labels': torch.cat([payloads[client_id][split_name]['labels'] for client_id in client_ids], dim=0),
    }


def _summary_from_payloads(
    dataset_name: str,
    raw_root: Path,
    payloads: dict[str, dict[str, dict[str, torch.Tensor]]],
    *,
    seed: int,
    val_ratio: float,
    class_names: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Build a JSON-ready summary for one prepared image dataset."""

    first_client = next(iter(payloads.values()))
    example_shape = list(first_client["train"]["images"].shape[1:])
    server_payloads = {
        split_name: _combine_server_payload(payloads, split_name)
        for split_name in ('train', 'val', 'test')
    }
    return {
        "dataset": str(dataset_name).lower(),
        "raw_root": str(raw_root),
        "num_clients": len(payloads),
        "split_seed": int(seed),
        "val_ratio": float(val_ratio),
        "num_classes": 0 if class_names is None else len(class_names),
        "class_names": None if class_names is None else list(class_names),
        "image_shape": example_shape,
        "clients": {
            client_id: {
                split_name: {
                    "rows": int(split_payload["labels"].shape[0]),
                    "label_histogram": torch.bincount(split_payload["labels"], minlength=(0 if class_names is None else len(class_names))).tolist(),
                }
                for split_name, split_payload in client_payload.items()
            }
            for client_id, client_payload in payloads.items()
        },
        "server": {
            split_name: {
                "rows": int(split_payload['labels'].shape[0]),
                "label_histogram": torch.bincount(split_payload['labels'], minlength=(0 if class_names is None else len(class_names))).tolist(),
            }
            for split_name, split_payload in server_payloads.items()
        },
    }


def save_client_split_payloads(
    output_dir: Path,
    dataset_name: str,
    payloads: dict[str, dict[str, dict[str, torch.Tensor]]],
    *,
    seed: int,
    val_ratio: float,
    raw_root: Path,
    class_names: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Persist per-client split payloads and return the preparation summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    server_dir = output_dir / 'server'
    server_dir.mkdir(parents=True, exist_ok=True)
    for client_id, client_payload in payloads.items():
        client_dir = output_dir / "clients" / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_payload in client_payload.items():
            torch.save(split_payload, client_dir / f"{split_name}.pt")
    for split_name in ('train', 'val', 'test'):
        torch.save(_combine_server_payload(payloads, split_name), server_dir / f'{split_name}.pt')
    summary = _summary_from_payloads(
        dataset_name,
        raw_root,
        payloads,
        seed=seed,
        val_ratio=val_ratio,
        class_names=class_names,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def prepare_image_classification_dataset(
    dataset_name: str,
    raw_root: Path,
    output_dir: Path,
    *,
    num_clients: int = 3,
    val_ratio: float = 0.1,
    seed: int = 2026,
) -> dict[str, Any]:
    """Download one dataset, split it into client shards, and persist the result."""

    dataset_key = str(dataset_name).lower()
    train_images, train_labels, class_names = _load_torchvision_dataset(dataset_key, raw_root, train=True)
    test_images, test_labels, _ = _load_torchvision_dataset(dataset_key, raw_root, train=False)
    reset_output_dir(output_dir)
    payloads = build_client_split_payloads(
        train_images,
        train_labels,
        test_images,
        test_labels,
        dataset_name=dataset_key,
        num_clients=num_clients,
        val_ratio=val_ratio,
        seed=seed,
    )
    return save_client_split_payloads(
        output_dir,
        dataset_key,
        payloads,
        seed=seed,
        val_ratio=val_ratio,
        raw_root=raw_root,
        class_names=class_names,
    )


def main() -> None:
    """Parse CLI arguments and prepare MNIST/CIFAR-10 client shards."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "all"), default="all")
    parser.add_argument("--raw-root", default="../data/raw_image_datasets")
    parser.add_argument("--output-root", default="../data")
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    datasets_to_prepare = [args.dataset] if args.dataset != "all" else ["mnist", "cifar10"]
    summary = {}
    for dataset_name in datasets_to_prepare:
        summary[dataset_name] = prepare_image_classification_dataset(
            dataset_name,
            raw_root / dataset_name,
            output_root / dataset_name,
            num_clients=args.num_clients,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
