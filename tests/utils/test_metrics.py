import torch

from federated_ts.utils.metrics import mae, mape, mse


def test_forecasting_metrics():
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 2.0])
    assert mse(pred, target) == 0.5
    assert mae(pred, target) == 0.5
    assert mape(torch.tensor([2.0]), torch.tensor([1.0])) == 100.0
