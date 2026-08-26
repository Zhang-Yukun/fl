import json
from pathlib import Path

import pytest
import torch

import fedlab.federated.methods.encoded as encoded_methods
from fedlab.federated.algorithms import load_captured_update_records, run_centralized, run_federated


def _write_split(root: Path, client_id: str, split: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    client_dir = root / 'clients' / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': images, 'labels': labels}, client_dir / f'{split}.pt')


def _prepare_classification_split_dir(
    root: Path,
    *,
    client_ids: list[str],
    image_shape: tuple[int, int, int],
    num_classes: int,
) -> None:
    train_images_all = []
    train_labels_all = []
    val_images_all = []
    val_labels_all = []
    test_images_all = []
    test_labels_all = []
    for client_offset, client_id in enumerate(client_ids):
        base_value = float(client_offset) / 10.0
        train_images = torch.full((6, *image_shape), base_value, dtype=torch.float32)
        train_labels = torch.tensor([index % num_classes for index in range(6)], dtype=torch.long)
        val_images = torch.full((3, *image_shape), base_value + 0.1, dtype=torch.float32)
        val_labels = torch.tensor([index % num_classes for index in range(3)], dtype=torch.long)
        test_images = torch.full((3, *image_shape), base_value + 0.2, dtype=torch.float32)
        test_labels = torch.tensor([index % num_classes for index in range(3)], dtype=torch.long)
        _write_split(root, client_id, 'train', train_images, train_labels)
        _write_split(root, client_id, 'val', val_images, val_labels)
        _write_split(root, client_id, 'test', test_images, test_labels)
        train_images_all.append(train_images)
        train_labels_all.append(train_labels)
        val_images_all.append(val_images)
        val_labels_all.append(val_labels)
        test_images_all.append(test_images)
        test_labels_all.append(test_labels)
    server_dir = root / 'server'
    server_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': torch.cat(train_images_all, dim=0), 'labels': torch.cat(train_labels_all, dim=0)}, server_dir / 'train.pt')
    torch.save({'images': torch.cat(val_images_all, dim=0), 'labels': torch.cat(val_labels_all, dim=0)}, server_dir / 'val.pt')
    torch.save({'images': torch.cat(test_images_all, dim=0), 'labels': torch.cat(test_labels_all, dim=0)}, server_dir / 'test.pt')
    (root / 'summary.json').write_text(
        json.dumps({'class_names': [f'class_{index}' for index in range(num_classes)]}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


class _IdentityEgaCodec(torch.nn.Module):
    """Minimal codec that makes EGA tests deterministic without pretraining artifacts."""

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = int(block_size)
        self.encoded_dim = int(block_size)
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        return blocks.to(torch.float32)

    def decode_blocks(self, encoded_blocks: torch.Tensor) -> torch.Tensor:
        return encoded_blocks.to(torch.float32)


def _patch_identity_ega_codec(monkeypatch, *, block_size: int = 8) -> None:
    def _fake_load_ega_codec(config, device, num_clients, allow_pretrain):
        del config, device, num_clients, allow_pretrain
        return _IdentityEgaCodec(block_size=block_size)

    def _fake_load_ega_codec_payload(config, payload, device, num_clients):
        del config, payload, device, num_clients
        return _IdentityEgaCodec(block_size=block_size)

    monkeypatch.setattr(encoded_methods, 'load_ega_codec', _fake_load_ega_codec)
    monkeypatch.setattr(encoded_methods, 'load_ega_codec_payload', _fake_load_ega_codec_payload)


def _classification_config(
    split_dir: Path,
    output_dir: Path,
    *,
    client_ids: list[str],
    image_shape: tuple[int, int, int],
    num_classes: int,
    algorithm: str = 'fedavg',
) -> dict:
    config = {
        'experiment': {'output_dir': str(output_dir), 'mode': 'federated'},
        'runtime': {'device': 'cpu', 'log_level': 'INFO', 'deterministic': True, 'seed': 2026},
        'task': {'type': 'classification'},
        'data': {
            'split_dir': str(split_dir),
            'clients': client_ids,
            'batch_size': 2,
            'shuffle_train': False,
            'num_workers': 0,
            'image_shape': list(image_shape),
            'num_classes': num_classes,
        },
        'model': {'name': 'small_cnn', 'hidden_channels': 4, 'dropout': 0.0},
        'training': {'lr': 0.001, 'optimizer': 'adam', 'loss': 'cross_entropy', 'patience': 1, 'min_delta': 0.0},
        'centralized': {'rounds': 1},
        'federated': {'algorithm': algorithm, 'rounds': 1, 'local_epochs': 1},
        'transport': {'upload_mode': 'update', 'download_mode': 'model'},
        'attack': {'enabled': False, 'target_type': 'update_payload', 'frequency_rounds': 1, 'steps': 1, 'async_enabled': False, 'device': 'same'},
        'tracking': {'enabled': False},
        'evaluation': {'metrics': ['accuracy']},
        'artifacts': {'config_formats': ['yaml'], 'save_every_rounds': 0},
    }
    if algorithm == 'sparse_fedavg':
        config['federated']['topk_fraction'] = 1.0
    if algorithm == 'ega_fedavg':
        config['federated']['quantization_seed'] = 2026
        config['ega'] = {
            'artifact_path': str(output_dir / 'artifacts' / 'ega_codec.pt'),
            'block_size': 8,
            'encoded_dim': 8,
            'hidden_dim': 8,
            'residual_blocks': 0,
            'quantization_level': 64,
            'encoded_dtype': 'float32',
            'download_method': 'ega',
            'download_dtype': 'float32',
            'error_feedback': False,
            'min_normalization': 1e-6,
        }
    return config


@pytest.mark.parametrize(
    ('client_ids', 'image_shape'),
    [
        (['m1', 'm2', 'm3'], (1, 4, 4)),
        (['c1', 'c2', 'c3'], (3, 8, 8)),
    ],
)
def test_centralized_run_supports_classification_task(tmp_path, client_ids, image_shape):
    split_dir = tmp_path / 'split'
    output_dir = tmp_path / 'centralized'
    _prepare_classification_split_dir(split_dir, client_ids=client_ids, image_shape=image_shape, num_classes=3)
    config = _classification_config(split_dir, output_dir, client_ids=client_ids, image_shape=image_shape, num_classes=3)

    summary = run_centralized(config)

    assert 'accuracy' in summary
    assert (output_dir / 'metrics.json').exists()
    assert (output_dir / 'summary.json').exists()


@pytest.mark.parametrize('algorithm', ['fedavg', 'sparse_fedavg', 'ega_fedavg'])
@pytest.mark.parametrize(
    ('client_ids', 'image_shape'),
    [
        (['m1', 'm2', 'm3'], (1, 4, 4)),
        (['c1', 'c2', 'c3'], (3, 8, 8)),
    ],
)
def test_federated_run_supports_classification_task(tmp_path, monkeypatch, algorithm, client_ids, image_shape):
    split_dir = tmp_path / f'split_{algorithm}'
    output_dir = tmp_path / f'output_{algorithm}'
    _prepare_classification_split_dir(split_dir, client_ids=client_ids, image_shape=image_shape, num_classes=3)
    config = _classification_config(
        split_dir,
        output_dir,
        client_ids=client_ids,
        image_shape=image_shape,
        num_classes=3,
        algorithm=algorithm,
    )
    if algorithm == 'ega_fedavg':
        _patch_identity_ega_codec(monkeypatch)

    summary = run_federated(config)
    captures = load_captured_update_records(output_dir)

    assert 'accuracy' in summary['test']
    assert summary['attack_evaluations'] == 0
    assert len(captures) == len(client_ids)
    assert {record['client_id'] for record in captures} == set(client_ids)
    assert (output_dir / 'saved_updates' / 'index.json').exists()


def test_federated_run_supports_classification_attacks(tmp_path):
    _prepare_classification_split_dir(tmp_path, client_ids=['m1', 'm2', 'm3'], image_shape=(1, 4, 4), num_classes=3)
    output_dir = tmp_path / 'output_attack'
    config = _classification_config(
        tmp_path,
        output_dir,
        client_ids=['m1', 'm2', 'm3'],
        image_shape=(1, 4, 4),
        num_classes=3,
        algorithm='fedavg',
    )
    config['attack'].update({
        'enabled': True,
        'target_type': 'update_payload',
        'frequency_rounds': 1,
        'sample_count': 1,
        'max_samples': 1,
        'clients_per_round': 1,
        'client_selection': 'first',
        'steps': 1,
        'optimizer': 'adam',
        'local_optimizer': 'adam',
        'async_enabled': False,
        'device': 'cpu',
    })

    summary = run_federated(config)

    assert summary['attack_evaluations'] == 2
    assert summary['attack_target_type'] == 'update_payload'
    assert summary['attack_primary_metric_name'] == 'budget_recovered_fraction'
    assert (output_dir / 'attack_results.json').exists()
    assert sorted((output_dir / 'attack_artifacts').rglob('*.pt'))
