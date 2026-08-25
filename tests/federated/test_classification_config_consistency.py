from pathlib import Path

import pytest

from fedlab.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize(
    ("config_name", "dataset_dir", "client_ids", "algorithm"),
    [
        ("mnist_classification.yaml", "../data/mnist", ["m1", "m2", "m3"], "fedavg"),
        ("mnist_topk_classification.yaml", "../data/mnist", ["m1", "m2", "m3"], "sparse_fedavg"),
        ("mnist_ega_classification.yaml", "../data/mnist", ["m1", "m2", "m3"], "ega_fedavg"),
        ("cifar10_classification.yaml", "../data/cifar10", ["c1", "c2", "c3"], "fedavg"),
        ("cifar10_topk_classification.yaml", "../data/cifar10", ["c1", "c2", "c3"], "sparse_fedavg"),
        ("cifar10_ega_classification.yaml", "../data/cifar10", ["c1", "c2", "c3"], "ega_fedavg"),
    ],
)
def test_image_classification_configs_match_expected_dataset_and_algorithm(config_name, dataset_dir, client_ids, algorithm):
    config = load_config(CONFIG_DIR / config_name)

    assert config['task']['type'] == 'classification'
    assert config['data']['split_dir'] == dataset_dir
    assert config['data']['clients'] == client_ids
    assert config['federated']['algorithm'] == algorithm
    assert config['training']['loss'] == 'cross_entropy'
    assert config['evaluation']['metrics'] == ['accuracy']


def test_classification_algorithm_configs_only_define_relevant_blocks():
    mnist_topk = load_config(CONFIG_DIR / 'mnist_topk_classification.yaml')
    mnist_ega = load_config(CONFIG_DIR / 'mnist_ega_classification.yaml')
    cifar_topk = load_config(CONFIG_DIR / 'cifar10_topk_classification.yaml')
    cifar_ega = load_config(CONFIG_DIR / 'cifar10_ega_classification.yaml')

    assert mnist_topk['federated']['topk_fraction'] == pytest.approx(0.10)
    assert cifar_topk['federated']['topk_fraction'] == pytest.approx(0.10)
    assert 'ega' not in mnist_topk
    assert 'ega' not in cifar_topk
    assert 'topk_fraction' not in mnist_ega['federated']
    assert 'topk_fraction' not in cifar_ega['federated']
    assert 'ega' in mnist_ega
    assert 'ega' in cifar_ega
