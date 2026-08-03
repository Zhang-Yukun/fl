from collections import OrderedDict

import pytest
import torch

from federated_ts.utils.aggregation import fedaware_weights


def test_fedaware_weights_return_simplex_and_adapt():
    updates = [
        OrderedDict(weight=torch.tensor([1.0, 0.0])),
        OrderedDict(weight=torch.tensor([0.0, 1.0])),
        OrderedDict(weight=torch.tensor([1.0, 1.0])),
    ]
    weights = fedaware_weights(updates, [1.0, 1.0, 4.0], alpha=1.0, steps=80, lr=0.2)

    assert pytest.approx(sum(weights), rel=1e-6, abs=1e-6) == 1.0
    assert all(weight >= 0.0 for weight in weights)
    assert weights[2] < (4.0 / 6.0)
