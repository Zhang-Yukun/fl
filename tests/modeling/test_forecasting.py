import torch

from federated_ts.modeling.forecasting import build_model


def test_mlp_forecaster_output_shape():
    model = build_model({"data": {"seq_len": 4, "pred_len": 2}, "model": {"name": "mlp", "channels": 1, "hidden_size": 8}})
    assert model(torch.zeros(3, 4, 1)).shape == (3, 2, 1)


def test_patchtst_forecaster_output_shape():
    model = build_model({
        "data": {"seq_len": 21, "pred_len": 7},
        "model": {"name": "patchtst", "channels": 1, "patch_len": 7, "stride": 4, "d_model": 16, "n_heads": 4, "e_layers": 1, "d_ff": 32},
    })
    assert model(torch.zeros(2, 21, 1)).shape == (2, 7, 1)


def test_fedpetuning_patchtst_reduces_trainable_parameters():
    model = build_model({
        "data": {"seq_len": 21, "pred_len": 7},
        "federated": {"algorithm": "fedpetuning"},
        "model": {
            "name": "patchtst",
            "channels": 1,
            "patch_len": 7,
            "stride": 4,
            "d_model": 16,
            "n_heads": 4,
            "e_layers": 1,
            "d_ff": 32,
            "peft": {"enabled": True, "method": "fedpetuning", "bottleneck_dim": 4, "train_head": True},
        },
    })
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert model(torch.zeros(2, 21, 1)).shape == (2, 7, 1)
    assert 0 < trainable < total
