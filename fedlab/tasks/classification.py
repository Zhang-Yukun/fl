"""Image classification task plugin backed by pre-split client tensors."""

from __future__ import annotations

from torch import nn

from fedlab.datasets.image_classification import build_federated_image_classification_loaders
from fedlab.modeling.classification import build_model as build_classification_model
from fedlab.tasks.base import TaskSpec
from fedlab.tasks.registry import register_task
from fedlab.utils.metrics import accuracy


def _classification_metrics(pred, target):
    return {
        'accuracy': accuracy(pred, target),
    }


CLASSIFICATION_TASK = TaskSpec(
    name='classification',
    build_model=build_classification_model,
    build_federated_loaders=build_federated_image_classification_loaders,
    create_loss=lambda _config: nn.CrossEntropyLoss(),
    compute_metrics=_classification_metrics,
    default_loss='cross_entropy',
    default_metrics=('accuracy',),
    default_optimizer='adam',
    primary_metric='accuracy',
    primary_metric_mode='max',
)


register_task(
    CLASSIFICATION_TASK,
    'image_classification',
    'mnist_classification',
    'cifar10_classification',
)
