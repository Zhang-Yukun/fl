import pytest

from fedlab.datasets import build_federated_loaders
from fedlab.federated.algorithms import is_compressed_algorithm
from fedlab.federated.client import FederatedClient
from fedlab.federated.methods import MethodConfigSpec, build_method, get_registered_method, is_registered_compressed, list_registered_methods
from fedlab.federated.server import FederatedServer
from fedlab.utils.config import load_config
from fedlab.federated.methods.base import FederatedMethod


EXPECTED_METHODS = {
    'adaptive_clipped_rdp_fedavg': False,
    'compressed_fedavg': True,
    'dp_topk_fedavg': True,
    'ega_fedavg': False,
    'fedavg': False,
    'fedaware': False,
    'qsgd_fedavg': False,
    'randomk_fedavg': True,
    'secure_quantized_fedavg': False,
    'sign_fedavg': False,
    'soteriafl': True,
    'sparse_fedavg': True,
}

IMPLEMENTED_METHODS = set(EXPECTED_METHODS)


EXPECTED_CONFIG_SPECS = {
    'fedavg': MethodConfigSpec(),
    'fedaware': MethodConfigSpec(root_blocks=frozenset({'fedaware'})),
    'adaptive_clipped_rdp_fedavg': MethodConfigSpec(root_blocks=frozenset({'adaptive_clipped_rdp'})),
    'compressed_fedavg': MethodConfigSpec(federated_keys=frozenset({'topk_fraction'})),
    'sparse_fedavg': MethodConfigSpec(federated_keys=frozenset({'topk_fraction'})),
    'dp_topk_fedavg': MethodConfigSpec(federated_keys=frozenset({'topk_fraction'}), uses_privacy_block=True),
    'randomk_fedavg': MethodConfigSpec(federated_keys=frozenset({'topk_fraction', 'randomk_seed'})),
    'soteriafl': MethodConfigSpec(federated_keys=frozenset({'topk_fraction', 'randomk_seed'}), uses_privacy_block=True),
    'secure_quantized_fedavg': MethodConfigSpec(
        federated_keys=frozenset({'quantization_dtype', 'quantization_stochastic_rounding', 'quantization_seed'}),
        uses_privacy_block=True,
    ),
    'sign_fedavg': MethodConfigSpec(),
    'qsgd_fedavg': MethodConfigSpec(federated_keys=frozenset({'qsgd_levels', 'quantization_seed'})),
    'ega_fedavg': MethodConfigSpec(federated_keys=frozenset({'quantization_seed'}), root_blocks=frozenset({'ega'})),
}


def test_method_registry_covers_current_algorithm_names():
    registered = {item.name: item.compressed for item in list_registered_methods()}

    assert registered == EXPECTED_METHODS


@pytest.mark.parametrize(('name', 'compressed'), sorted(EXPECTED_METHODS.items()))
def test_registered_method_exposes_expected_capabilities(name, compressed):
    item = get_registered_method(name)
    method = build_method(name)

    assert item.compressed is compressed
    assert is_registered_compressed(name) is compressed
    assert isinstance(method, FederatedMethod)
    assert method.name == name
    assert method.capabilities.compressed is compressed
    assert method.capabilities.implemented is (name in IMPLEMENTED_METHODS)


def test_registered_methods_expose_config_metadata():
    for name, expected in EXPECTED_CONFIG_SPECS.items():
        item = get_registered_method(name)
        method = build_method(name)

        assert item.config_spec == expected
        assert method.config_spec == expected


def test_runtime_compressed_resolution_matches_registry():
    for name, compressed in EXPECTED_METHODS.items():
        config = {'federated': {'algorithm': name}}
        assert is_compressed_algorithm(config) is compressed


def test_client_and_server_bind_registered_method(tmp_path):
    config = load_config('configs/test.yaml', ['federated.algorithm=fedavg'])
    config['experiment']['output_dir'] = str(tmp_path)
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    client_id, train_loader = next(iter(train_loaders.items()))
    total_train_samples = sum(len(loader.dataset) for loader in train_loaders.values())

    client = FederatedClient(
        client_id,
        train_loader,
        config,
        device='cpu',
        total_train_samples=total_train_samples,
        total_clients=len(train_loaders),
    )
    server = FederatedServer(config, val_loader, test_loader, device='cpu')

    assert client.method.name == 'fedavg'
    assert server.method.name == 'fedavg'
    assert client.method.capabilities.compressed is False
    assert server.method.capabilities.compressed is False
