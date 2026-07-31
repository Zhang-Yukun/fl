import torch

from federated_ts.modeling.forecasting import build_model
from federated_ts.engine.training import first_batch_gradient
from federated_ts.security.attacks import dlg_attack, idlg_attack, summarize_attack_results
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
        "attack": {
            "steps": 1,
            "lr": 0.01,
            "success_mse_threshold": 0.01,
            "success_rate_threshold": 0.03,
            "data_range": 1.0,
            "model_mode": "eval",
            "seed": 7,
        },
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
    for result in (dlg, idlg):
        assert result.mse == result.reconstruction_mse
        assert torch.isfinite(torch.tensor(result.psnr))
        assert torch.isfinite(torch.tensor(result.ssim))
        assert result.iterations == 1
        assert result.time_seconds >= 0.0
        assert result.success_threshold == 0.01
        record = result.to_record()
        assert record["mse"] == record["reconstruction_mse"]
        assert {"psnr", "ssim", "iterations", "time_seconds", "gradient_mse"} <= set(record)

    summary = summarize_attack_results([dlg, idlg], success_rate_threshold=0.03)
    assert set(summary["methods"]) == {"DLG", "iDLG"}
    assert summary["success_rate_threshold"] == 0.03


def test_attack_gradient_sampling_supports_eval_mode():
    config = _tiny_patchtst_config()
    config["model"]["dropout"] = 0.5
    device = torch.device("cpu")
    x = torch.randn(2, 8, 1)
    y = torch.randn(2, 2, 1)
    loader = [(x, y)]

    torch.manual_seed(11)
    model_a = build_model(config).to(device)
    grads_a, sample_x_a, _ = first_batch_gradient(model_a, loader, device, max_samples=1, model_mode="eval")
    torch.manual_seed(11)
    model_b = build_model(config).to(device)
    grads_b, sample_x_b, _ = first_batch_gradient(model_b, loader, device, max_samples=1, model_mode="eval")

    assert torch.allclose(sample_x_a, sample_x_b)
    assert all(torch.allclose(left, right) for left, right in zip(grads_a, grads_b))
