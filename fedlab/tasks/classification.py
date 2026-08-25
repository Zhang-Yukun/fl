"""Image classification task plugin backed by pre-split client tensors."""

from __future__ import annotations

from torch import nn

from fedlab.datasets.image_classification import build_federated_image_classification_loaders
from fedlab.modeling.classification import build_model as build_classification_model
from fedlab.tasks.base import TaskSpec
from fedlab.utils.metrics import accuracy, cross_entropy


def _classification_metrics(pred, target):
    ce = cross_entropy(pred, target)
    acc = accuracy(pred, target)
    error_rate = 1.0 - acc
    return {
        'cross_entropy': ce,
        'accuracy': acc,
        'mse': ce,
        'mae': error_rate,
        'mape': error_rate * 100.0,
    }


CLASSIFICATION_TASK = TaskSpec(
    name='classification',
    build_model=build_classification_model,
    build_federated_loaders=build_federated_image_classification_loaders,
    create_loss=lambda _config: nn.CrossEntropyLoss(),
    compute_metrics=_classification_metrics,
    default_loss='cross_entropy',
    default_metrics=('cross_entropy', 'accuracy'),
    default_optimizer='adam',
    primary_metric='cross_entropy',
    primary_metric_mode='min',
)
