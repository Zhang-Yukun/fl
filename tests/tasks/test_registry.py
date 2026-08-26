import torch

from fedlab.datasets import build_federated_loaders
from fedlab.modeling import build_model
from fedlab.tasks import build_optimizer, compute_metrics, create_loss, get_model_task, get_task, metric_names, optimizer_name, primary_metric, primary_metric_mode
from fedlab.tasks.base import TaskSpec
from fedlab.tasks.registry import list_registered_tasks, register_task, task_plugin
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

    assert optimizer.__class__.__name__ == "Adam"
    assert optimizer_name(config) == "adam"


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


def test_classification_task_resolves_accuracy_and_cross_entropy():
    config = {
        'task': {'type': 'classification'},
        'data': {'image_shape': [1, 4, 4], 'num_classes': 3},
        'model': {'name': 'small_cnn', 'hidden_channels': 4},
        'training': {'loss': 'cross_entropy', 'optimizer': 'adam', 'lr': 0.001},
        'evaluation': {'metrics': ['accuracy', 'cross_entropy']},
    }

    task = get_task(config)
    model = build_model(config)
    metrics = compute_metrics(config, torch.tensor([[2.0, 0.0, -1.0]]), torch.tensor([0]))
    optimizer = build_optimizer([torch.nn.Parameter(torch.ones(1))], config)

    assert task.name == 'classification'
    assert get_model_task(model).name == 'classification'
    assert metric_names(config) == ('accuracy', 'cross_entropy')
    assert metrics['accuracy'] == 1.0
    assert 'cross_entropy' in metrics
    assert optimizer.__class__.__name__ == 'Adam'


def test_builtin_task_aliases_are_registered():
    registered = list_registered_tasks()

    assert registered['forecasting'].name == 'forecasting'
    assert registered['rare_earth_forecasting'].name == 'forecasting'
    assert registered['classification'].name == 'classification'
    assert registered['mnist_classification'].name == 'classification'


def test_register_task_supports_aliases(monkeypatch):
    from fedlab.tasks import registry as task_registry

    snapshot = list_registered_tasks()
    monkeypatch.setattr(task_registry, '_TASKS', dict(snapshot))
    monkeypatch.setattr(task_registry, '_BUILTIN_TASKS_LOADED', True)

    dummy_task = TaskSpec(
        name='dummy',
        build_model=lambda _config: torch.nn.Identity(),
        build_federated_loaders=lambda _config: ({}, None, None),
    )

    register_task(dummy_task, 'dummy_alias')

    assert task_registry._TASKS['dummy'] is dummy_task
    assert task_registry._TASKS['dummy_alias'] is dummy_task
    assert get_task({'task': {'type': 'dummy_alias'}}) is dummy_task


def test_task_plugin_rejects_conflicting_alias(monkeypatch):
    from fedlab.tasks import registry as task_registry

    snapshot = list_registered_tasks()
    monkeypatch.setattr(task_registry, '_TASKS', dict(snapshot))
    monkeypatch.setattr(task_registry, '_BUILTIN_TASKS_LOADED', True)

    conflicting = TaskSpec(
        name='classification_conflict',
        build_model=lambda _config: torch.nn.Identity(),
        build_federated_loaders=lambda _config: ({}, None, None),
    )

    try:
        register_task(conflicting, 'classification')
    except ValueError as exc:
        assert 'classification' in str(exc)
    else:
        raise AssertionError('Expected conflicting alias registration to fail')


def test_task_plugin_decorator_registers_task(monkeypatch):
    from fedlab.tasks import registry as task_registry

    snapshot = list_registered_tasks()
    monkeypatch.setattr(task_registry, '_TASKS', dict(snapshot))
    monkeypatch.setattr(task_registry, '_BUILTIN_TASKS_LOADED', True)

    plugin_task = TaskSpec(
        name='decorated_task',
        build_model=lambda _config: torch.nn.Identity(),
        build_federated_loaders=lambda _config: ({}, None, None),
    )

    task_plugin('decorated_alias')(plugin_task)

    assert task_registry._TASKS['decorated_task'] is plugin_task
    assert task_registry._TASKS['decorated_alias'] is plugin_task
