import pytest
import torch

from fedlab.modeling import build_model
from fedlab.engine.training import first_batch_gradient, first_batch_sample
from fedlab.security.attacks import AttackResult, apply_set_recovery_metrics, dlg_attack, idlg_attack, save_attack_artifacts, summarize_attack_results
from fedlab.security.registry import register_attack_summary_metric, register_recovery_metric
from fedlab.utils.serialization import serialize_model, subtract_state


def _tiny_patchtst_config(target_type: str = "update_payload"):
    return {
        "runtime": {"device": "cpu"},
        "data": {"seq_len": 8, "pred_len": 2},
        "training": {"lr": 0.001},
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
            "target_type": target_type,
            "report_metrics": "auto",
            "steps": 1,
            "lr": 0.05,
            "optimizer": "lbfgs",
            "restarts": 1,
            "lbfgs_history_size": 5,
            "input_clip": 5.0,
            "target_clip": 5.0,
            "success_mse_threshold": 0.5,
            "success_rate_threshold": 0.03,
            "data_range": 1.0,
            "model_mode": "eval",
            "local_optimizer": "adam",
            "local_lr": 0.001,
            "seed": 7,
        },
    }


def _tiny_classification_config(target_type: str = "update_payload"):
    return {
        "runtime": {"device": "cpu"},
        "task": {"type": "classification"},
        "data": {"image_shape": [1, 4, 4], "num_classes": 3, "dataset_name": "mnist"},
        "training": {"lr": 0.01, "loss": "cross_entropy", "optimizer": "adam"},
        "model": {"name": "small_cnn", "hidden_channels": 4, "dropout": 0.0},
        "attack": {
            "target_type": target_type,
            "report_metrics": "auto",
            "steps": 2,
            "lr": 0.05,
            "optimizer": "adam",
            "restarts": 1,
            "input_clip": 5.0,
            "target_clip": 5.0,
            "success_mse_threshold": 0.5,
            "success_rate_threshold": 0.03,
            "data_range": 1.0,
            "model_mode": "eval",
            "local_optimizer": "adam",
            "local_lr": 0.01,
            "seed": 7,
        },
        "evaluation": {"metrics": ["cross_entropy", "accuracy"]},
    }


def test_gradient_attacks_run_on_vendored_patchtst():
    config = _tiny_patchtst_config(target_type="gradient")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 8, 1)
    y = torch.randn(1, 2, 1)
    loss = torch.nn.functional.mse_loss(model(x), y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    state = serialize_model(model)

    dlg = dlg_attack(config, state, [grad.detach() for grad in grads], x, y, device, target_type="gradient")
    idlg = idlg_attack(config, state, [grad.detach() for grad in grads], x, y, device, target_type="gradient")

    assert torch.isfinite(torch.tensor(dlg.reconstruction_mse))
    assert torch.isfinite(torch.tensor(idlg.reconstruction_mse))
    for result in (dlg, idlg):
        assert result.mse == result.reconstruction_mse
        assert result.target_type == "gradient"
        assert torch.isfinite(torch.tensor(result.psnr))
        assert torch.isfinite(torch.tensor(result.ssim))
        assert result.iterations == 1
        assert result.time_seconds >= 0.0
        assert result.success_threshold == 0.5
        record = result.to_record()
        assert record["primary_metric_name"] == result.metric_name
        assert record["primary_metric_value"] == result.mse
        assert {"psnr", "ssim", "iterations", "time_seconds", "objective_mse", "primary_metric_name", "primary_metric_value", "target_type"} <= set(record)
        assert "mse" not in record
        assert "gradient_mse" not in record

    dlg.client_id = "Nd2O3"
    idlg.client_id = "CeO2"
    summary = summarize_attack_results([dlg, idlg], success_rate_threshold=0.03)
    assert set(summary["methods"]) == {"DLG", "iDLG"}
    assert summary["primary_metric_name"] == "reconstruction_mse"
    assert summary["primary_metric_direction"] == "higher_is_more_private"
    assert summary["target_type"] == "gradient"
    assert summary["overall_avg_primary_metric_value"] is not None
    assert summary["success_rate_threshold"] == 0.03
    assert summary["methods"]["DLG"]["primary_metric_name"] == "reconstruction_mse"
    assert summary["methods"]["DLG"]["total_count"] == 1
    assert "avg_objective_mse" in summary["methods"]["DLG"]
    assert set(summary["clients"]) == {"Nd2O3", "CeO2"}
    assert summary["clients"]["Nd2O3"]["methods"]["DLG"]["total_count"] == 1


def test_attack_summary_uses_registered_primary_metric_direction(monkeypatch):
    import fedlab.security.registry as registry_module

    snapshot = registry_module.list_registered_attack_summary_metrics()
    monkeypatch.setattr(registry_module, '_ATTACK_SUMMARY_METRICS', dict(snapshot))
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACK_SUMMARY_METRICS_LOADED', True)

    register_attack_summary_metric(
        'custom_privacy',
        lambda result: getattr(result, 'custom_privacy', None),
        average_key='overall_avg_custom_privacy',
        best_key='overall_best_custom_privacy',
        best_objective='max',
        privacy_direction='lower_is_more_private',
    )

    result = AttackResult(
        name='DLG',
        mse=0.0,
        psnr=0.0,
        ssim=0.0,
        iterations=1,
        time_seconds=0.0,
        success=False,
        success_threshold=0.5,
        gradient_mse=0.0,
        target_type='gradient',
        metric_name='custom_privacy',
    )
    result.custom_privacy = 0.8

    summary = summarize_attack_results([result], success_rate_threshold=0.03)

    assert summary['primary_metric_direction'] == 'lower_is_more_private'
    assert summary['overall_avg_primary_metric_value'] == 0.8
    assert summary['overall_avg_custom_privacy'] == 0.8
    assert summary['methods']['DLG']['avg_custom_privacy'] == 0.8


def test_update_payload_attacks_run_on_vendored_patchtst():
    config = _tiny_patchtst_config(target_type="update_payload")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 8, 1)
    y = torch.randn(1, 2, 1)
    state = serialize_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    target_update = subtract_state(serialize_model(model), state)

    reference_inputs = torch.cat([x, x + 5.0], dim=0)
    dlg = dlg_attack(config, state, target_update, x, y, device, target_type="update_payload", reference_inputs=reference_inputs)
    idlg = idlg_attack(config, state, target_update, x, y, device, target_type="update_payload", reference_inputs=reference_inputs)

    assert dlg.target_type == "update_payload"
    assert idlg.target_type == "update_payload"
    assert torch.isfinite(torch.tensor(dlg.reconstruction_mse))
    assert torch.isfinite(torch.tensor(idlg.reconstruction_mse))
    assert dlg.metric_name == "nearest_client_train_mse"
    assert idlg.metric_name == "nearest_client_train_mse"
    assert dlg.nearest_client_train_mse is not None
    assert idlg.exact_target_mse is None
    assert dlg.reference_label == "nearest_client_train"
    assert torch.allclose(dlg.reference_x, x)
    assert torch.allclose(dlg.reference_y, y)

    dlg.client_id = "Nd2O3"
    idlg.client_id = "Nd2O3"
    summary = summarize_attack_results([dlg, idlg], success_rate_threshold=0.03)
    assert summary["target_type"] == "update_payload"
    assert summary["primary_metric_name"] == "nearest_client_train_mse"
    assert summary["methods"]["DLG"]["target_type"] == "update_payload"
    assert summary["overall_avg_nearest_client_train_mse"] is not None
    assert summary["overall_avg_objective_mse"] is not None
    assert summary["clients"]["Nd2O3"]["methods"]["DLG"]["target_type"] == "update_payload"


def test_first_batch_sample_can_span_multiple_batches_when_max_samples_exceeds_batch_size():
    device = torch.device("cpu")
    loader = [
        (torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
        (torch.tensor([[2.0], [3.0]]), torch.tensor([2, 3])),
        (torch.tensor([[4.0], [5.0]]), torch.tensor([4, 5])),
    ]

    x, y = first_batch_sample(loader, device, max_samples=5, batch_index=0)

    assert x.shape[0] == 5
    assert torch.equal(x.squeeze(-1).cpu(), torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(y.cpu(), torch.tensor([0, 1, 2, 3, 4]))



def test_attack_gradient_sampling_supports_eval_mode():
    config = _tiny_patchtst_config(target_type="gradient")
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



def test_attack_artifacts_persist_reconstructions(tmp_path):
    config = _tiny_patchtst_config(target_type="gradient")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 8, 1)
    y = torch.randn(1, 2, 1)
    loss = torch.nn.functional.mse_loss(model(x), y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    state = serialize_model(model)

    result = dlg_attack(config, state, [grad.detach() for grad in grads], x, y, device, target_type="gradient")
    result.client_id = "Nd2O3"
    result.round_index = 2
    result.sample_index = 0

    records = save_attack_artifacts(tmp_path, [result])

    assert len(records) == 1
    assert records[0]["artifact_path"] is not None
    artifact_path = tmp_path / records[0]["artifact_path"]
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert torch.allclose(payload["real_x"], x)
    assert torch.allclose(payload["real_y"], y)
    assert payload["reference_x"] is not None
    assert payload["reference_label"] is not None
    assert payload["reconstructed_x"].shape == x.shape
    assert payload["reconstructed_y"] is not None


def test_apply_set_recovery_metrics_supports_custom_registered_metric(monkeypatch):
    import fedlab.security.registry as registry_module

    snapshot = registry_module.list_registered_recovery_metrics()
    monkeypatch.setattr(registry_module, '_RECOVERY_METRICS', dict(snapshot))
    monkeypatch.setattr(registry_module, '_BUILTIN_RECOVERY_METRICS_LOADED', True)

    def custom_matrix(reconstructed, reference_inputs, _data_range):
        recon = reconstructed.detach().cpu().float().reshape(reconstructed.shape[0], -1)
        refs = reference_inputs.detach().cpu().float().reshape(reference_inputs.shape[0], -1)
        return torch.mean(torch.abs(recon[:, None, :] - refs[None, :, :]), dim=-1)

    register_recovery_metric(
        'l1',
        custom_matrix,
        default_objective='min',
        default_threshold=lambda _config, _data_range: 0.01,
    )

    config = _tiny_patchtst_config(target_type='update_payload')
    config['attack']['recovery_match_metric'] = 'l1'
    config['attack']['recovery_success_metric'] = 'l1'
    results = [
        AttackResult(
            name='DLG',
            mse=0.0,
            psnr=0.0,
            ssim=0.0,
            iterations=1,
            time_seconds=0.0,
            success=False,
            success_threshold=0.5,
            gradient_mse=0.0,
            target_type='update_payload',
            reconstructed_x=torch.tensor([[[0.0], [0.0]]]),
            reconstructed_y=torch.tensor([[[0.0]]]),
        )
    ]

    apply_set_recovery_metrics(
        results,
        reference_inputs=torch.tensor([[[0.0], [0.0]], [[1.0], [1.0]]]),
        reference_targets=torch.tensor([[[0.0]], [[1.0]]]),
        config=config,
    )

    assert results[0].matched_reference_metric_name == 'l1'
    assert results[0].budget_recovered_fraction == 1.0


def test_apply_set_recovery_metrics_uses_one_to_one_matching_for_time_series_and_classification():
    ts_config = _tiny_patchtst_config(target_type="update_payload")
    ts_config["attack"]["recovery_success_threshold"] = 0.01
    ts_results = [
        AttackResult(
            name="DLG",
            mse=0.0,
            psnr=0.0,
            ssim=0.0,
            iterations=1,
            time_seconds=0.0,
            success=False,
            success_threshold=0.5,
            gradient_mse=0.0,
            target_type="update_payload",
            reconstructed_x=torch.tensor([[[0.0], [0.0]]]),
            reconstructed_y=torch.tensor([[[0.0]]]),
        ),
        AttackResult(
            name="DLG",
            mse=0.0,
            psnr=0.0,
            ssim=0.0,
            iterations=1,
            time_seconds=0.0,
            success=False,
            success_threshold=0.5,
            gradient_mse=0.0,
            target_type="update_payload",
            reconstructed_x=torch.tensor([[[0.0], [0.0]]]),
            reconstructed_y=torch.tensor([[[0.0]]]),
        ),
    ]
    ts_reference_inputs = torch.tensor([[[0.0], [0.0]], [[1.0], [1.0]]])
    ts_reference_targets = torch.tensor([[[0.0]], [[1.0]]])
    apply_set_recovery_metrics(ts_results, reference_inputs=ts_reference_inputs, reference_targets=ts_reference_targets, config=ts_config)
    assert ts_results[0].metric_name == "budget_recovered_fraction"
    assert ts_results[0].budget_recovered_fraction == pytest.approx(0.5)
    assert ts_results[0].coverage_recovered_fraction == pytest.approx(0.5)
    assert ts_results[0].matched_reference_indices == [0]
    assert ts_results[1].matched_reference_indices == [1]
    assert ts_results[0].reference_label == "matched_client_train"
    assert ts_results[0].success is True
    assert ts_results[1].success is False

    cls_config = _tiny_classification_config(target_type="update_payload")
    cls_config["attack"]["recovery_success_threshold"] = 0.01
    cls_results = [
        AttackResult(
            name="iDLG",
            mse=0.0,
            psnr=0.0,
            ssim=0.0,
            iterations=1,
            time_seconds=0.0,
            success=False,
            success_threshold=0.5,
            gradient_mse=0.0,
            target_type="update_payload",
            reconstructed_x=torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
            reconstructed_y=torch.tensor([1]),
        ),
    ]
    cls_reference_inputs = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 1.0], [1.0, 0.0]]]])
    cls_reference_targets = torch.tensor([1, 2], dtype=torch.long)
    apply_set_recovery_metrics(cls_results, reference_inputs=cls_reference_inputs, reference_targets=cls_reference_targets, config=cls_config)
    assert cls_results[0].budget_recovered_fraction == pytest.approx(1.0)
    assert cls_results[0].coverage_recovered_fraction == pytest.approx(0.5)
    assert cls_results[0].reference_y is not None
    assert torch.equal(cls_results[0].reference_y, torch.tensor([1]))



def test_classification_attacks_support_integer_labels_and_logits():
    config = _tiny_classification_config(target_type="gradient")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 1, 4, 4)
    y = torch.tensor([2], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    state = serialize_model(model)

    dlg = dlg_attack(config, state, [grad.detach() for grad in grads], x, y, device, target_type="gradient")
    idlg = idlg_attack(config, state, [grad.detach() for grad in grads], x, y, device, target_type="gradient")

    assert dlg.reconstructed_x.shape == x.shape
    assert dlg.reconstructed_y.shape == (1, 3)
    assert idlg.reconstructed_y.shape == y.shape
    assert torch.isfinite(torch.tensor(dlg.reconstruction_mse))
    assert torch.isfinite(torch.tensor(idlg.reconstruction_mse))


def test_classification_idlg_inferrs_label_from_gradient_target():
    config = _tiny_classification_config(target_type="gradient")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 1, 4, 4)
    true_y = torch.tensor([2], dtype=torch.long)
    decoy_y = torch.tensor([0], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(model(x), true_y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    state = serialize_model(model)

    result = idlg_attack(config, state, [grad.detach() for grad in grads], x, decoy_y, device, target_type="gradient")

    assert result.reconstructed_y is not None
    assert torch.equal(result.reconstructed_y.cpu(), true_y)
    assert not torch.equal(result.reconstructed_y.cpu(), decoy_y)


def test_classification_idlg_inferrs_label_from_update_payload():
    config = _tiny_classification_config(target_type="update_payload")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 1, 4, 4)
    true_y = torch.tensor([1], dtype=torch.long)
    decoy_y = torch.tensor([0], dtype=torch.long)
    state = serialize_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), true_y)
    loss.backward()
    optimizer.step()
    target_update = subtract_state(serialize_model(model), state)

    result = idlg_attack(config, state, target_update, x, decoy_y, device, target_type="update_payload")

    assert result.reconstructed_y is not None
    assert torch.equal(result.reconstructed_y.cpu(), true_y)
    assert not torch.equal(result.reconstructed_y.cpu(), decoy_y)


def test_classification_idlg_does_not_reuse_probe_labels_when_label_inference_is_invalid():
    config = _tiny_classification_config(target_type="gradient")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(2, 1, 4, 4)
    true_y = torch.tensor([2, 1], dtype=torch.long)
    decoy_y = torch.tensor([0, 0], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(model(x), true_y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    state = serialize_model(model)

    result = idlg_attack(config, state, [grad.detach() for grad in grads], x, decoy_y, device, target_type="gradient")

    assert result.reconstructed_y is not None
    assert result.reconstructed_y.shape == (2, 3)
    assert torch.is_floating_point(result.reconstructed_y)
    assert not torch.equal(result.reconstructed_y.detach().cpu().argmax(dim=1), decoy_y)


def test_classification_update_payload_idlg_does_not_reuse_probe_labels_for_multi_sample_batches():
    config = _tiny_classification_config(target_type="update_payload")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(2, 1, 4, 4)
    true_y = torch.tensor([1, 2], dtype=torch.long)
    decoy_y = torch.tensor([0, 0], dtype=torch.long)
    state = serialize_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), true_y)
    loss.backward()
    optimizer.step()
    target_update = subtract_state(serialize_model(model), state)

    result = idlg_attack(config, state, target_update, x, decoy_y, device, target_type="update_payload")

    assert result.reconstructed_y is not None
    assert result.reconstructed_y.shape == (2, 3)
    assert torch.is_floating_point(result.reconstructed_y)
    assert not torch.equal(result.reconstructed_y.detach().cpu().argmax(dim=1), decoy_y)



def test_time_series_idlg_does_not_reuse_probe_targets_for_update_payloads():
    config = _tiny_patchtst_config(target_type="update_payload")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 8, 1)
    true_y = torch.randn(1, 2, 1)
    decoy_y = true_y + 10.0
    state = serialize_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), true_y)
    loss.backward()
    optimizer.step()
    target_update = subtract_state(serialize_model(model), state)

    result = idlg_attack(config, state, target_update, x, decoy_y, device, target_type="update_payload")

    assert result.reconstructed_y is not None
    assert result.reconstructed_y.shape == true_y.shape
    assert not torch.allclose(result.reconstructed_y.cpu(), decoy_y)



def test_classification_update_payload_attacks_keep_reference_labels():
    config = _tiny_classification_config(target_type="update_payload")
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(1, 1, 4, 4)
    y = torch.tensor([1], dtype=torch.long)
    state = serialize_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    target_update = subtract_state(serialize_model(model), state)

    reference_inputs = torch.cat([x, x + 0.5], dim=0)
    reference_targets = torch.tensor([1, 2], dtype=torch.long)
    dlg = dlg_attack(
        config,
        state,
        target_update,
        x,
        y,
        device,
        target_type="update_payload",
        reference_inputs=reference_inputs,
        reference_targets=reference_targets,
    )

    assert dlg.metric_name == "nearest_client_train_mse"
    assert dlg.reference_label == "nearest_client_train"
    assert dlg.reference_x.shape == x.shape
    assert dlg.nearest_client_train_indices is not None
    expected_reference_y = reference_targets.index_select(0, torch.tensor(dlg.nearest_client_train_indices, dtype=torch.long))
    assert dlg.reference_y is not None
    assert torch.equal(dlg.reference_y, expected_reference_y)
    assert dlg.reconstructed_y is not None
