import numpy as np
import torch

from federated_ts.data import Standardizer, WindowDataset, split_array
from federated_ts.metrics import mae, mape, mse
from federated_ts.models import build_model


def test_standardizer_and_window_dataset():
    values = np.arange(20, dtype="float32")[:, None]
    train, val, test = split_array(values, 0.5, 0.25)
    assert len(train) == 10
    assert len(val) == 5
    assert len(test) == 5
    scaler = Standardizer.fit(train)
    restored = scaler.inverse_transform(scaler.transform(train))
    np.testing.assert_allclose(restored, train, rtol=1e-6, atol=1e-6)
    dataset = WindowDataset(values, seq_len=4, pred_len=2)
    x, y = dataset[0]
    assert x.shape == (4, 1)
    assert y.shape == (2, 1)


def test_metrics_and_model_output_shape():
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 2.0])
    assert mse(pred, target) == 0.5
    assert mae(pred, target) == 0.5
    assert mape(torch.tensor([2.0]), torch.tensor([1.0])) == 100.0
    model = build_model({"data": {"seq_len": 4, "pred_len": 2}, "model": {"name": "mlp", "channels": 1, "hidden_size": 8}})
    assert model(torch.zeros(3, 4, 1)).shape == (3, 2, 1)
