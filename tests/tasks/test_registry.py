import torch

from fedlab.datasets import build_federated_loaders
from fedlab.modeling import build_model
from fedlab.tasks import build_optimizer, compute_metrics, create_loss, get_model_task, get_task, metric_names, optimizer_name, primary_metric, primary_metric_mode
from fedlab.utils.config import load_config


def test_default_config_resolves_forecasting_task():
    config = load_config('configs/test.yaml')

    task = get_task(config)

    assert task.name == 'forecasting'
    assert primary_metric(config) == 'mse'
    assert primary_metric_mode(config) == 'min'


def test_default_optimizer_falls_back_to_sgd_for_forecasting():
    config = {"task": {"type": "forecasting"}, "training": {"lr": 0.01}}

    optimizer = build_optimizer([torch.nn.Parameter(torch.ones(1))], config)

    assert optimizer.__class__.__name__ == "SGD"
    assert optimizer_name(config) == "sgd"


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


def test_component_registry_allows_config_driven_loss_metric_and_optimizer():
    config = {
        'task': {'type': 'forecasting'},
        'training': {'loss': 'mae', 'optimizer': 'sgd', 'lr': 0.01, 'momentum': 0.0},
        'evaluation': {'metrics': ['mae']},
    }

    loss_fn = create_loss(config)
    optimizer = build_optimizer([torch.nn.Parameter(torch.ones(1))], config)
    metrics = compute_metrics(config, torch.tensor([1.0]), torch.tensor([2.0]))

    assert loss_fn.__class__.__name__ == 'L1Loss'
    assert optimizer.__class__.__name__ == 'SGD'
    assert metric_names(config) == ('mae', 'mse', 'mape')
    assert metrics['mae'] == 1.0
    assert 'mse' in metrics
    assert optimizer_name(config) == 'sgd'
