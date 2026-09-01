import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

from fedlab.federated.algorithms import run_federated


TOOLS_DIR = Path(__file__).parents[2] / 'fedlab' / 'tools'
SCRIPT_PATH = TOOLS_DIR / 'replay_saved_model_evaluation.py'


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = _load_module(SCRIPT_PATH, 'replay_saved_model_evaluation')


def _write_split(root: Path, client_id: str, split: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    client_dir = root / 'clients' / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': images, 'labels': labels}, client_dir / f'{split}.pt')


def _prepare_classification_split_dir(root: Path, *, client_ids: list[str], image_shape: tuple[int, int, int], num_classes: int) -> None:
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


def _classification_config(split_dir: Path, output_dir: Path) -> dict:
    return {
        'experiment': {'output_dir': str(output_dir), 'mode': 'federated'},
        'runtime': {'device': 'cpu', 'log_level': 'INFO', 'deterministic': True, 'seed': 2026},
        'task': {'type': 'classification'},
        'data': {
            'split_dir': str(split_dir),
            'clients': ['m1', 'm2', 'm3'],
            'batch_size': 2,
            'shuffle_train': False,
            'num_workers': 0,
            'image_shape': [1, 4, 4],
            'num_classes': 3,
        },
        'model': {'name': 'small_cnn', 'hidden_channels': 4, 'dropout': 0.0},
        'training': {'epochs': 1, 'lr': 0.001, 'optimizer': 'adam', 'loss': 'cross_entropy', 'patience': 1, 'min_delta': 0.0},
        'federated': {'algorithm': 'fedavg', 'rounds': 1},
        'attack': {'enabled': False, 'target_type': 'update_payload', 'frequency_rounds': 1, 'steps': 1, 'async_enabled': False, 'device': 'same'},
        'tracking': {'enabled': False},
        'evaluation': {'metrics': ['accuracy']},
        'artifacts': {'config_formats': ['yaml'], 'save_every_rounds': 0},
    }


def _run_script(model_path: Path, output_dir: Path, *extra_args: str) -> dict[str, object]:
    argv = ['replay_saved_model_evaluation', str(model_path), '--output-dir', str(output_dir), *extra_args]
    stdout = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = argv
        with redirect_stdout(stdout):
            module.main()
    finally:
        sys.argv = old_argv
    return json.loads(stdout.getvalue())


def _prepare_forecasting_split_dir(root: Path, *, client_ids: list[str]) -> None:
    for client_offset, client_id in enumerate(client_ids):
        client_dir = root / 'clients' / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        for split, start, length in [('train', 0, 40), ('val', 40, 20), ('test', 60, 20)]:
            offset = client_offset * 100
            dates = [f'2020-01-{((index + offset) % 28) + 1:02d}' for index in range(start, start + length)]
            values = [float(offset + index) for index in range(start, start + length)]
            rows = 'date,value\n' + '\n'.join(f'{date},{value}' for date, value in zip(dates, values)) + '\n'
            (client_dir / f'{split}.csv').write_text(rows, encoding='utf-8')


def _forecasting_config(split_dir: Path, output_dir: Path) -> dict:
    return {
        'experiment': {'output_dir': str(output_dir), 'mode': 'federated'},
        'runtime': {'device': 'cpu', 'log_level': 'INFO', 'deterministic': True, 'seed': 2026},
        'task': {'type': 'forecasting'},
        'data': {
            'split_dir': str(split_dir),
            'clients': ['Nd2O3', 'CeO2', 'La2O3'],
            'seq_len': 4,
            'pred_len': 2,
            'batch_size': 8,
            'shuffle_train': False,
            'num_workers': 0,
        },
        'model': {'name': 'lstm', 'input_dim': 1, 'hidden_dim': 8, 'num_layers': 1, 'dropout': 0.0, 'pred_len': 2},
        'training': {'epochs': 1, 'lr': 0.001, 'optimizer': 'adam', 'loss': 'mse', 'patience': 1, 'min_delta': 0.0},
        'federated': {'algorithm': 'fedavg', 'rounds': 1},
        'attack': {'enabled': False, 'target_type': 'update_payload', 'frequency_rounds': 1, 'steps': 1, 'async_enabled': False, 'device': 'same'},
        'tracking': {'enabled': False},
        'evaluation': {'metrics': ['mse', 'mae', 'mape']},
        'artifacts': {'config_formats': ['yaml'], 'save_every_rounds': 0},
    }


def _prepare_forecasting_test_only_split(source_root: Path, target_root: Path, *, client_ids: list[str]) -> None:
    for client_id in client_ids:
        source = source_root / 'clients' / client_id / 'test.csv'
        client_dir = target_root / 'clients' / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        (client_dir / 'test.csv').write_text(source.read_text(encoding='utf-8'), encoding='utf-8')


def test_replay_saved_model_evaluation_matches_online_test_results(tmp_path):
    split_dir = tmp_path / 'split'
    online_dir = tmp_path / 'online'
    replay_dir = tmp_path / 'replay'
    _prepare_classification_split_dir(split_dir, client_ids=['m1', 'm2', 'm3'], image_shape=(1, 4, 4), num_classes=3)
    config = _classification_config(split_dir, online_dir)

    summary = run_federated(config)
    payload = _run_script(online_dir / 'model.pt', replay_dir)

    replay_metrics = json.loads((replay_dir / 'test_metrics.json').read_text(encoding='utf-8'))
    replay_summary = json.loads((replay_dir / 'test_summary.json').read_text(encoding='utf-8'))

    assert payload['test_metrics_path'] == str(replay_dir / 'test_metrics.json')
    assert payload['test_summary_path'] == str(replay_dir / 'test_summary.json')
    assert replay_metrics == pytest.approx(summary['test'])
    assert replay_summary['test'] == pytest.approx(summary['test'])
    assert replay_summary['protocol_test'] == pytest.approx(summary['protocol_test'])
    assert replay_summary['evaluation_mode'] == 'offline_saved_model_test'
    assert replay_summary['mode'] == 'federated'
    assert replay_summary['task_type'] == 'classification'


def test_replay_saved_model_evaluation_script_exists():
    assert SCRIPT_PATH.exists()


def test_replay_saved_model_evaluation_rare_uses_saved_scalers_with_test_only_data(tmp_path):
    split_dir = tmp_path / 'rare_full'
    online_dir = tmp_path / 'online_rare'
    replay_data_dir = tmp_path / 'rare_test_only'
    replay_dir = tmp_path / 'replay_rare'
    _prepare_forecasting_split_dir(split_dir, client_ids=['Nd2O3', 'CeO2', 'La2O3'])
    _prepare_forecasting_test_only_split(split_dir, replay_data_dir, client_ids=['Nd2O3', 'CeO2', 'La2O3'])
    config = _forecasting_config(split_dir, online_dir)

    summary = run_federated(config)
    assert (online_dir / 'evaluation_context.json').exists()

    payload = _run_script(
        online_dir / 'model.pt',
        replay_dir,
        '--config', str(online_dir / 'config.yaml'),
        '--data-dir', str(replay_data_dir),
    )

    replay_metrics = json.loads((replay_dir / 'test_metrics.json').read_text(encoding='utf-8'))
    replay_summary = json.loads((replay_dir / 'test_summary.json').read_text(encoding='utf-8'))

    assert replay_metrics == pytest.approx(summary['test'])
    assert replay_summary['test'] == pytest.approx(summary['test'])
    assert replay_summary['protocol_test'] == pytest.approx(summary['protocol_test'])
    assert replay_summary['source_evaluation_context_path'] == str(online_dir / 'evaluation_context.json')
    assert payload['test_summary_path'] == str(replay_dir / 'test_summary.json')
