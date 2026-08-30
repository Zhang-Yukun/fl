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
    if 'mnist/' in config_name:
        assert config['model']['name'] == 'medium_cnn'
        assert config['model']['hidden_channels'] == 32
    else:
        assert config['model']['name'] == 'medium_cnn'
        assert config['model']['hidden_channels'] == 32


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
    if 'mnist/' in config_name:
        assert config['model']['name'] == 'medium_cnn'
        assert config['model']['hidden_channels'] == 32
    else:
        assert config['model']['name'] == 'medium_cnn'
        assert config['model']['hidden_channels'] == 32



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
    assert config['attack']['frequency_rounds'] == 30
    assert config['attack']['methods'] == ['dlg', 'idlg']
    assert config['tracking']['enabled'] is True
    assert config['tracking']['offline'] is True
    assert config['artifacts']['config_formats'] == ['yaml', 'json']


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



def test_all_task_ega_configs_share_the_same_common_preset_without_stale_keys():
    rare = load_config(CONFIG_DIR / 'rare/ega.yaml')['ega']
    mnist = load_config(CONFIG_DIR / 'mnist/ega.yaml')['ega']
    cifar = load_config(CONFIG_DIR / 'cifar10/ega.yaml')['ega']

    assert rare == mnist == cifar
    assert rare['artifact_path'] == 'artifacts/ega/ega_h240_v1.pt'
    assert rare['block_size'] == 256
    assert rare['encoded_dim'] == 168
    assert rare['hidden_dim'] == 2048
    assert rare['residual_blocks'] == 4
    assert rare['quantization_level'] == 159
    assert rare['normalization_ema'] == 0.98
    assert rare['pretrain']['epochs'] == 220
    assert rare['pretrain']['patience'] == 44
    assert rare['pretrain']['lr'] == 0.0002
    assert rare['pretrain']['train_groups'] == 50000
    assert rare['pretrain']['val_groups'] == 25000
    assert rare['pretrain']['batch_size'] == 128
    assert rare['pretrain']['seed'] == 2026
    assert rare['pretrain']['device'] == 'same'
    assert 'download_dtype' not in rare
    assert 'download_method' not in rare
    assert 'download_predictive_coding' not in rare
    assert 'download_stochastic_rounding' not in rare
    assert 'download_trainable_only' not in rare
    assert 'download_encoded_dtype' not in rare
