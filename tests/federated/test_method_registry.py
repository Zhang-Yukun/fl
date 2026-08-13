import pytest

from fedlab.federated.methods import build_method, get_registered_method, is_registered_compressed, list_registered_methods
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
