import torch

from federated_ts.modeling.forecasting import build_model


def test_mlp_forecaster_output_shape():
    model = build_model({"data": {"seq_len": 4, "pred_len": 2}, "model": {"name": "mlp", "channels": 1, "hidden_size": 8}})
    assert model(torch.zeros(3, 4, 1)).shape == (3, 2, 1)
