import torch

from fedlab.modeling import build_model


def test_small_cnn_classifier_output_shape():
    model = build_model({
        'task': {'type': 'classification'},
        'data': {'dataset_name': 'mnist', 'image_shape': [1, 28, 28], 'num_classes': 10},
        'model': {'name': 'small_cnn', 'hidden_channels': 8},
    })
    assert model(torch.zeros(4, 1, 28, 28)).shape == (4, 10)


def test_medium_cnn_classifier_output_shape_for_mnist():
    model = build_model({
        'task': {'type': 'classification'},
        'data': {'dataset_name': 'mnist', 'image_shape': [1, 28, 28], 'num_classes': 10},
        'model': {'name': 'medium_cnn', 'hidden_channels': 8},
    })
    assert model(torch.zeros(4, 1, 28, 28)).shape == (4, 10)


def test_large_cnn_classifier_output_shape_for_cifar10():
    model = build_model({
        'task': {'type': 'classification'},
        'data': {'dataset_name': 'cifar10', 'image_shape': [3, 32, 32], 'num_classes': 10},
        'model': {'name': 'large_cnn', 'hidden_channels': 8},
    })
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_flatten_classifier_output_shape():
    model = build_model({
        'task': {'type': 'classification'},
        'data': {'image_shape': [3, 32, 32], 'num_classes': 5},
        'model': {'name': 'mlp', 'hidden_size': 16},
    })
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 5)
