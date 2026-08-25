import torch

from fedlab.utils.metrics import accuracy, cross_entropy, mae, mape, mse


def test_forecasting_metrics():
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 2.0])
    assert mse(pred, target) == 0.5
    assert mae(pred, target) == 0.5
    assert mape(torch.tensor([2.0]), torch.tensor([1.0])) == 100.0


def test_classification_metrics():
    logits = torch.tensor([[3.0, 1.0], [0.1, 0.9]])
    target = torch.tensor([0, 1])
    assert accuracy(logits, target) == 1.0
    assert cross_entropy(logits, target) > 0.0
