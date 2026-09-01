import importlib.util
import json
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[2] / 'fedlab' / 'tools' / 'prepare_role_datasets.py'
spec = importlib.util.spec_from_file_location('prepare_role_datasets_script', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


prepare_rare_role_dataset = module.prepare_rare_role_dataset
prepare_classification_role_dataset = module.prepare_classification_role_dataset
prepare_role_datasets = module.prepare_role_datasets


def _write_rare_split(root: Path, client_id: str, split: str, values: list[int]) -> None:
    client_dir = root / 'clients' / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    rows = 'date,value\n' + '\n'.join(f'2020-01-{index + 1:02d},{value}' for index, value in enumerate(values)) + '\n'
    (client_dir / f'{split}.csv').write_text(rows, encoding='utf-8')


def _prepare_rare_split_dir(root: Path) -> None:
    summary_rows = {}
    for client_offset, client_id in enumerate(['Nd2O3', 'CeO2', 'La2O3']):
        train_values = [client_offset * 100 + index for index in range(10)]
        val_values = [client_offset * 100 + 10 + index for index in range(4)]
        test_values = [client_offset * 100 + 20 + index for index in range(4)]
        _write_rare_split(root, client_id, 'train', train_values)
        _write_rare_split(root, client_id, 'val', val_values)
        _write_rare_split(root, client_id, 'test', test_values)
        summary_rows[client_id] = {
            'train': len(train_values),
            'val': len(val_values),
            'test': len(test_values),
        }
    (root / 'summary.json').write_text(json.dumps({'rows': summary_rows}, ensure_ascii=False, indent=2), encoding='utf-8')


def _write_image_split(root: Path, client_id: str, split: str, fill_value: float, labels: list[int]) -> None:
    client_dir = root / 'clients' / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'images': torch.full((len(labels), 1, 2, 2), fill_value, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
    }
    torch.save(payload, client_dir / f'{split}.pt')


def _prepare_mnist_split_dir(root: Path) -> None:
    for client_offset, client_id in enumerate(['m1', 'm2', 'm3']):
        _write_image_split(root, client_id, 'train', float(client_offset), [0, 1, 2])
        _write_image_split(root, client_id, 'val', float(client_offset) + 0.1, [0, 1])
        _write_image_split(root, client_id, 'test', float(client_offset) + 0.2, [2, 1])
    server_dir = root / 'server'
    server_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': torch.zeros(6, 1, 2, 2), 'labels': torch.tensor([0, 1, 2, 0, 1, 2])}, server_dir / 'val.pt')
    torch.save({'images': torch.ones(6, 1, 2, 2), 'labels': torch.tensor([2, 1, 0, 2, 1, 0])}, server_dir / 'test.pt')
    (root / 'summary.json').write_text(
        json.dumps({'dataset': 'mnist', 'class_names': ['0', '1', '2']}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def test_prepare_rare_role_dataset_materializes_minimal_role_dirs(tmp_path):
    split_dir = tmp_path / 'rare_earth_rawdata2'
    output_dir = tmp_path / 'roles'
    _prepare_rare_split_dir(split_dir)
    evaluation_context = tmp_path / 'evaluation_context.json'
    evaluation_context.write_text(json.dumps({'clients': {'Nd2O3': {'scale_mean': [1.0], 'scale_std': [2.0]}}}), encoding='utf-8')

    summary = prepare_rare_role_dataset(
        split_dir,
        output_dir,
        attack_clients=['Nd2O3'],
        evaluation_context_path=evaluation_context,
    )

    assert (output_dir / 'client' / 'Nd2O3' / 'clients' / 'Nd2O3' / 'train.csv').exists()
    assert not (output_dir / 'client' / 'Nd2O3' / 'clients' / 'Nd2O3' / 'val.csv').exists()
    assert (output_dir / 'server' / 'clients' / 'CeO2' / 'val.csv').exists()
    assert (output_dir / 'server' / 'clients' / 'La2O3' / 'test.csv').exists()
    assert (output_dir / 'attack' / 'Nd2O3' / 'clients' / 'Nd2O3' / 'train.csv').exists()
    assert not (output_dir / 'attack' / 'CeO2').exists()
    assert (output_dir / 'test' / 'clients' / 'Nd2O3' / 'test.csv').exists()
    assert (output_dir / 'test' / 'evaluation_context.json').exists()
    assert summary['notes']['evaluation_context_path'] == str(output_dir / 'test' / 'evaluation_context.json')
    saved_summary = json.loads((output_dir / 'summary.json').read_text(encoding='utf-8'))
    assert saved_summary['task'] == 'rare'


def test_prepare_classification_role_dataset_materializes_minimal_role_dirs(tmp_path):
    split_dir = tmp_path / 'mnist'
    output_dir = tmp_path / 'roles'
    _prepare_mnist_split_dir(split_dir)

    summary = prepare_classification_role_dataset(
        split_dir,
        output_dir,
        task='mnist',
        attack_clients=['m2'],
    )

    assert (output_dir / 'client' / 'm1' / 'clients' / 'm1' / 'train.pt').exists()
    assert not (output_dir / 'client' / 'm1' / 'clients' / 'm1' / 'val.pt').exists()
    assert (output_dir / 'server' / 'server' / 'val.pt').exists()
    assert (output_dir / 'server' / 'server' / 'test.pt').exists()
    assert (output_dir / 'attack' / 'm2' / 'clients' / 'm2' / 'train.pt').exists()
    assert not (output_dir / 'attack' / 'm1').exists()
    assert (output_dir / 'test' / 'server' / 'test.pt').exists()
    payload = torch.load(output_dir / 'attack' / 'm2' / 'clients' / 'm2' / 'train.pt', map_location='cpu', weights_only=False)
    assert payload['labels'].tolist() == [0, 1, 2]
    assert summary['task'] == 'mnist'


def test_prepare_role_datasets_supports_all_tasks(tmp_path):
    source_root = tmp_path / 'data'
    output_root = tmp_path / 'role_datasets'
    _prepare_rare_split_dir(source_root / 'rare_earth_rawdata2')
    _prepare_mnist_split_dir(source_root / 'mnist')
    _prepare_mnist_split_dir(source_root / 'cifar10')

    summary = prepare_role_datasets(
        task='all',
        source_root=source_root,
        output_root=output_root,
    )

    assert set(summary) == {'rare', 'mnist', 'cifar10'}
    assert (output_root / 'rare' / 'server' / 'clients' / 'Nd2O3' / 'val.csv').exists()
    assert (output_root / 'mnist' / 'server' / 'server' / 'val.pt').exists()
    assert (output_root / 'cifar10' / 'test' / 'server' / 'test.pt').exists()
