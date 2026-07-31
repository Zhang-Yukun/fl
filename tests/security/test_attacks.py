import torch

from federated_ts.modeling.forecasting import build_model
from federated_ts.security.attacks import dlg_attack, idlg_attack
from federated_ts.utils.serialization import serialize_model


def _tiny_patchtst_config():
    return {
        "runtime": {"device": "cpu"},
        "data": {"seq_len": 8, "pred_len": 2},
        "model": {
            "name": "patchtst",
            "channels": 1,
            "patch_len": 4,
            "stride": 2,
            "d_model": 8,
            "n_heads": 1,
            "e_layers": 1,
            "d_ff": 16,
            "factor": 1,
            "activation": "gelu",
            "dropout": 0.0,
        },
        "attack": {"steps": 1, "lr": 0.01, "success_mse_threshold": 1e-4},
    }


def test_gradient_attacks_run_on_vendored_patchtst():
    config = _tiny_patchtst_config()
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 8, 1)
    y = torch.randn(1, 2, 1)
    loss = torch.nn.functional.mse_loss(model(x), y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    state = serialize_model(model)

    dlg = dlg_attack(config, state, [grad.detach() for grad in grads], x, y, device)
    idlg = idlg_attack(config, state, [grad.detach() for grad in grads], x, y, device)

    assert torch.isfinite(torch.tensor(dlg.reconstruction_mse))
    assert torch.isfinite(torch.tensor(idlg.reconstruction_mse))
