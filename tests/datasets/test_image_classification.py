import json
from pathlib import Path

import torch

from fedlab.datasets import build_federated_loaders


def _write_client_split(root: Path, client_id: str, split: str, count: int, value: float) -> None:
    client_dir = root / 'clients' / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    images = torch.full((count, 1, 4, 4), value, dtype=torch.float32)
    labels = torch.arange(count, dtype=torch.long) % 3
    torch.save({'images': images, 'labels': labels}, client_dir / f'{split}.pt')


def test_build_image_classification_loaders_from_split_dir(tmp_path):
    for client_index, client_id in enumerate(['m1', 'm2', 'm3'], start=1):
        _write_client_split(tmp_path, client_id, 'train', 6, float(client_index))
        _write_client_split(tmp_path, client_id, 'val', 3, float(client_index + 10))
        _write_client_split(tmp_path, client_id, 'test', 3, float(client_index + 20))
    (tmp_path / 'summary.json').write_text(json.dumps({'class_names': [str(index) for index in range(3)]}), encoding='utf-8')

    config = {
        'task': {'type': 'classification'},
        'data': {
            'split_dir': str(tmp_path),
            'clients': ['m1', 'm2', 'm3'],
            'batch_size': 2,
            'shuffle_train': False,
        },
    }
    train_loaders, val_loader, test_loader = build_federated_loaders(config)

    assert set(train_loaders) == {'m1', 'm2', 'm3'}
    x, y = next(iter(train_loaders['m1']))
    assert x.shape == (2, 1, 4, 4)
    assert y.dtype == torch.long
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    assert getattr(val_loader, 'class_names', None) == ['0', '1', '2']
