import torch

from fedlab.modeling import build_model as build_any_model
from fedlab.modeling.forecasting import build_model


def test_mlp_forecaster_output_shape():
    model = build_model({"data": {"seq_len": 4, "pred_len": 2}, "model": {"name": "mlp", "channels": 1, "hidden_size": 8}})
    assert model(torch.zeros(3, 4, 1)).shape == (3, 2, 1)


def test_patchtst_forecaster_output_shape():
    model = build_model({
        "data": {"seq_len": 21, "pred_len": 7},
        "model": {"name": "patchtst", "channels": 1, "patch_len": 7, "stride": 4, "d_model": 16, "n_heads": 4, "e_layers": 1, "d_ff": 32},
    })
    assert model(torch.zeros(2, 21, 1)).shape == (2, 7, 1)



def test_patchtst_forecaster_state_dict_uses_raw_reference_names():
    model = build_model({
        "data": {"seq_len": 21, "pred_len": 7},
        "model": {"name": "patchtst", "channels": 1, "patch_len": 7, "stride": 4, "d_model": 16, "n_heads": 4, "e_layers": 1, "d_ff": 32},
    })
    keys = list(model.state_dict().keys())
    assert keys
    assert not any(key.startswith("model.") for key in keys)
    assert any(key.startswith("patch_embedding.") for key in keys)



def test_forecasting_uniform_init_override_changes_parameters():
    torch.manual_seed(0)
    default_model = build_any_model({
        'task': {'type': 'forecasting'},
        'data': {'seq_len': 4, 'pred_len': 2},
        'model': {'name': 'mlp', 'channels': 1, 'hidden_size': 8},
    })
    torch.manual_seed(0)
    override_model = build_any_model({
        'task': {'type': 'forecasting'},
        'data': {'seq_len': 4, 'pred_len': 2},
        'model': {'name': 'mlp', 'channels': 1, 'hidden_size': 8, 'init': 'uniform', 'init_uniform_low': -0.25, 'init_uniform_high': 0.25},
    })
    assert not torch.equal(default_model.net[1].weight, override_model.net[1].weight)
    assert float(torch.max(override_model.net[1].weight).item()) <= 0.25
    assert float(torch.min(override_model.net[1].weight).item()) >= -0.25
