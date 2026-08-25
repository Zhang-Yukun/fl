from pathlib import Path

import torch

from fedlab.federated.algorithms import run_federated


def _write_split(root: Path, client_id: str, split: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    client_dir = root / 'clients' / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': images, 'labels': labels}, client_dir / f'{split}.pt')


def _prepare_classification_split_dir(root: Path) -> None:
    for client_offset, client_id in enumerate(['client1', 'client2', 'client3']):
        base_value = float(client_offset) / 10.0
        train_images = torch.full((6, 1, 4, 4), base_value, dtype=torch.float32)
        train_labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
        val_images = torch.full((3, 1, 4, 4), base_value + 0.1, dtype=torch.float32)
        val_labels = torch.tensor([0, 1, 2], dtype=torch.long)
        test_images = torch.full((3, 1, 4, 4), base_value + 0.2, dtype=torch.float32)
        test_labels = torch.tensor([0, 1, 2], dtype=torch.long)
        _write_split(root, client_id, 'train', train_images, train_labels)
        _write_split(root, client_id, 'val', val_images, val_labels)
        _write_split(root, client_id, 'test', test_images, test_labels)


def test_federated_run_supports_classification_task(tmp_path):
    _prepare_classification_split_dir(tmp_path)
    config = {
        'experiment': {'output_dir': str(tmp_path / 'output'), 'mode': 'federated'},
        'runtime': {'device': 'cpu', 'log_level': 'INFO', 'deterministic': True, 'seed': 2026},
        'task': {'type': 'classification'},
        'data': {
            'split_dir': str(tmp_path),
            'clients': ['client1', 'client2', 'client3'],
            'batch_size': 2,
            'shuffle_train': False,
            'num_workers': 0,
            'image_shape': [1, 4, 4],
            'num_classes': 3,
        },
        'model': {'name': 'small_cnn', 'hidden_channels': 4, 'dropout': 0.0},
        'training': {'lr': 0.001, 'optimizer': 'adam', 'loss': 'cross_entropy', 'patience': 1, 'min_delta': 0.0},
        'centralized': {'rounds': 1},
        'federated': {'algorithm': 'fedavg', 'rounds': 1, 'local_epochs': 1},
        'transport': {'upload_mode': 'update', 'download_mode': 'model'},
        'attack': {'enabled': False, 'target_type': 'gradient', 'steps': 1, 'async_enabled': False, 'device': 'same'},
        'tracking': {'enabled': False},
        'evaluation': {'metrics': ['cross_entropy', 'accuracy']},
        'artifacts': {'config_formats': ['yaml'], 'save_every_rounds': 0},
    }

    summary = run_federated(config)

    assert 'accuracy' in summary['test']
    assert 'cross_entropy' in summary['test']
    assert summary['attack_evaluations'] == 0
