from pathlib import Path

import pytest

from fedlab.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize(
    ("config_name", "dataset_dir", "client_ids", "algorithm"),
    [
        ("mnist/fedavg.yaml", "../data/mnist", ["m1", "m2", "m3"], "fedavg"),
        ("mnist/topk.yaml", "../data/mnist", ["m1", "m2", "m3"], "sparse_fedavg"),
        ("mnist/ega.yaml", "../data/mnist", ["m1", "m2", "m3"], "ega_fedavg"),
        ("cifar10/fedavg.yaml", "../data/cifar10", ["c1", "c2", "c3"], "fedavg"),
        ("cifar10/topk.yaml", "../data/cifar10", ["c1", "c2", "c3"], "sparse_fedavg"),
        ("cifar10/ega.yaml", "../data/cifar10", ["c1", "c2", "c3"], "ega_fedavg"),
    ],
)
def test_image_classification_configs_match_expected_dataset_and_algorithm(config_name, dataset_dir, client_ids, algorithm):
    config = load_config(CONFIG_DIR / config_name)

    assert config['task']['type'] == 'classification'
    assert config['data']['split_dir'] == dataset_dir
    assert config['data']['clients'] == client_ids
    assert config['federated']['algorithm'] == algorithm
    assert config['training']['loss'] == 'cross_entropy'
    assert config['training']['epochs'] == 1
    assert config['evaluation']['metrics'] == ['accuracy']


@pytest.mark.parametrize(
    ("config_name", "dataset_dir", "client_ids"),
    [
        ("mnist/centralized.yaml", "../data/mnist", ["m1", "m2", "m3"]),
        ("cifar10/centralized.yaml", "../data/cifar10", ["c1", "c2", "c3"]),
    ],
)
def test_image_classification_centralized_configs_match_expected_dataset(config_name, dataset_dir, client_ids):
    config = load_config(CONFIG_DIR / config_name)

    assert config['experiment']['mode'] == 'centralized'
    assert config['task']['type'] == 'classification'
    assert config['data']['split_dir'] == dataset_dir
    assert config['data']['clients'] == client_ids
    assert config['training']['loss'] == 'cross_entropy'
    assert config['training']['epochs'] == 300
    assert config['evaluation']['metrics'] == ['accuracy']



@pytest.mark.parametrize(
    ("config_name", "dataset_name"),
    [
        ("mnist/fedavg.yaml", "mnist"),
        ("cifar10/fedavg.yaml", "cifar10"),
    ],
)
def test_image_classification_configs_load_data_from_dedicated_common_files(config_name, dataset_name):
    config = load_config(CONFIG_DIR / config_name)

    assert config['data']['dataset_name'] == dataset_name
    assert config['data']['batch_size'] == 128
    assert config['training']['epochs'] == 1
    assert config['training']['optimizer'] == 'adam'
    assert config['federated']['rounds'] == 300


def test_classification_algorithm_configs_only_define_relevant_blocks():
    mnist_topk = load_config(CONFIG_DIR / 'mnist/topk.yaml')
    mnist_ega = load_config(CONFIG_DIR / 'mnist/ega.yaml')
    cifar_topk = load_config(CONFIG_DIR / 'cifar10/topk.yaml')
    cifar_ega = load_config(CONFIG_DIR / 'cifar10/ega.yaml')

    assert mnist_topk['federated']['topk_fraction'] == pytest.approx(0.10)
    assert cifar_topk['federated']['topk_fraction'] == pytest.approx(0.10)
    assert 'ega' not in mnist_topk
    assert 'ega' not in cifar_topk
    assert 'topk_fraction' not in mnist_ega['federated']
    assert 'topk_fraction' not in cifar_ega['federated']
    assert 'ega' in mnist_ega
    assert 'ega' in cifar_ega


@pytest.mark.parametrize(
    ('wrapper_name', 'canonical_name'),
    [
        ('mnist_classification.yaml', 'mnist/fedavg.yaml'),
        ('mnist_topk_classification.yaml', 'mnist/topk.yaml'),
        ('mnist_ega_classification.yaml', 'mnist/ega.yaml'),
        ('mnist_centralized_classification.yaml', 'mnist/centralized.yaml'),
        ('cifar10_classification.yaml', 'cifar10/fedavg.yaml'),
        ('cifar10_topk_classification.yaml', 'cifar10/topk.yaml'),
        ('cifar10_ega_classification.yaml', 'cifar10/ega.yaml'),
        ('cifar10_centralized_classification.yaml', 'cifar10/centralized.yaml'),
    ],
)
def test_legacy_classification_config_wrappers_match_canonical_configs(wrapper_name, canonical_name):
    wrapper = load_config(CONFIG_DIR / wrapper_name)
    canonical = load_config(CONFIG_DIR / canonical_name)

    assert wrapper == canonical
