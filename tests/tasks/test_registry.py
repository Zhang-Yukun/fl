import torch

from federated_ts.datasets import build_federated_loaders
from federated_ts.modeling import build_model
from federated_ts.tasks import get_model_task, get_task, primary_metric, primary_metric_mode
from federated_ts.utils.config import load_config


def test_default_config_resolves_forecasting_task():
    config = load_config('configs/test.yaml')

    task = get_task(config)

    assert task.name == 'forecasting'
    assert primary_metric(config) == 'mse'
    assert primary_metric_mode(config) == 'min'


def test_task_aware_model_builder_annotates_model():
    config = {
        'task': {'type': 'forecasting'},
        'data': {'seq_len': 4, 'pred_len': 2},
        'model': {'name': 'mlp', 'channels': 1, 'hidden_size': 8},
    }

    model = build_model(config)

    assert get_model_task(model).name == 'forecasting'
    assert model(torch.zeros(2, 4, 1)).shape == (2, 2, 1)


def test_task_aware_dataset_builder_matches_existing_clients():
    config = load_config('configs/test.yaml')

    train_loaders, val_loader, test_loader = build_federated_loaders(config)

    assert set(train_loaders) == {'Nd2O3', 'CeO2', 'La2O3'}
    assert len(val_loader) > 0
    assert len(test_loader) > 0
