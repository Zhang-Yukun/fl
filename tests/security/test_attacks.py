import pytest
import torch

from fedlab.modeling import build_model
from fedlab.engine.training import first_batch_sample
from fedlab.security.attack_common import AttackResult, apply_set_recovery_metrics, save_attack_artifacts, summarize_attack_results
from fedlab.security.dlg import dlg_attack
from fedlab.security.idlg import idlg_attack
from fedlab.security.registry import register_attack_artifact_field, register_attack_record_field, register_attack_summary_metric, register_recovery_metric
from fedlab.utils.serialization import serialize_model, subtract_state


def _tiny_patchtst_config():
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
            "target_type": "update_payload",
            "steps": 1,
            "lr": 0.05,
            "optimizer": "adam",
            "restarts": 1,
            "input_clip": 5.0,
            "target_clip": 5.0,
            "success_rate_threshold": 0.05,
            "data_range": 1.0,
            "model_mode": "eval",
            "local_optimizer": "adam",
            "local_lr": 0.001,
            "seed": 7,
        },
    }


def _tiny_classification_config():
    return {
        "runtime": {"device": "cpu"},
        "task": {"type": "classification"},
        "data": {"image_shape": [1, 4, 4], "num_classes": 3, "dataset_name": "mnist"},
        "training": {"lr": 0.01, "loss": "cross_entropy", "optimizer": "adam"},
        "model": {"name": "small_cnn", "hidden_channels": 4, "dropout": 0.0},
        "attack": {
            "target_type": "update_payload",
            "steps": 2,
            "lr": 0.05,
            "optimizer": "adam",
            "restarts": 1,
            "input_clip": 5.0,
            "target_clip": 5.0,
            "success_rate_threshold": 0.05,
            "data_range": 1.0,
            "model_mode": "eval",
            "local_optimizer": "adam",
            "local_lr": 0.01,
            "seed": 7,
        },
        "evaluation": {"metrics": ["cross_entropy", "accuracy"]},
    }



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
        iterations=1,
        time_seconds=0.0,
        success=False,
        success_threshold=0.5,
        gradient_mse=0.0,
        target_type='update_payload',
        metric_name='custom_privacy',
    )
    result.custom_privacy = 0.8

    summary = summarize_attack_results([result], success_rate_threshold=0.05)

    assert summary['primary_metric_direction'] == 'lower_is_more_private'
    assert summary['overall_avg_primary_metric_value'] == 0.8
    assert summary['overall_avg_custom_privacy'] == 0.8
    assert summary['methods']['DLG']['avg_custom_privacy'] == 0.8


def test_update_payload_attacks_run_on_vendored_patchtst():
    config = _tiny_patchtst_config()
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
    assert torch.isfinite(torch.tensor(dlg.mse))
    assert torch.isfinite(torch.tensor(idlg.mse))
    assert dlg.metric_name == "objective_mse"
    assert idlg.metric_name == "objective_mse"
    assert dlg.nearest_client_train_mse is not None
    assert idlg.nearest_client_train_mse is not None
    assert dlg.reference_label == "nearest_client_train"
    assert torch.allclose(dlg.reference_x, x)
    assert dlg.reference_y is None

    dlg.client_id = "Nd2O3"
    idlg.client_id = "Nd2O3"
    apply_set_recovery_metrics([dlg, idlg], reference_inputs=reference_inputs, reference_targets=torch.cat([y, y + 5.0], dim=0), config=config)
    summary = summarize_attack_results([dlg, idlg], success_rate_threshold=0.05)
    assert summary["target_type"] == "update_payload"
    assert summary["overall_success_rate_threshold"] == 0.05
    assert summary["primary_metric_name"] == "budget_recovered_fraction"
    assert summary["methods"]["DLG"]["target_type"] == "update_payload"
    assert summary["overall_avg_budget_recovered_fraction"] is not None
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




def test_attack_serialization_supports_custom_registered_fields(tmp_path, monkeypatch):
    import fedlab.security.registry as registry_module

    registry_module.list_registered_attack_record_fields()
    registry_module.list_registered_attack_artifact_fields()
    snapshot_record_fields = dict(registry_module._ATTACK_RECORD_FIELDS)
    snapshot_record_order = list(registry_module._ATTACK_RECORD_FIELD_ORDER)
    snapshot_artifact_fields = dict(registry_module._ATTACK_ARTIFACT_FIELDS)
    snapshot_artifact_order = list(registry_module._ATTACK_ARTIFACT_FIELD_ORDER)
    monkeypatch.setattr(registry_module, '_ATTACK_RECORD_FIELDS', dict(snapshot_record_fields))
    monkeypatch.setattr(registry_module, '_ATTACK_RECORD_FIELD_ORDER', list(snapshot_record_order))
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACK_RECORD_FIELDS_LOADED', True)
    monkeypatch.setattr(registry_module, '_ATTACK_ARTIFACT_FIELDS', dict(snapshot_artifact_fields))
    monkeypatch.setattr(registry_module, '_ATTACK_ARTIFACT_FIELD_ORDER', list(snapshot_artifact_order))
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACK_ARTIFACT_FIELDS_LOADED', True)

    register_attack_record_field('custom_json_field', lambda result: getattr(result, 'custom_json_field', None))
    register_attack_artifact_field('custom_artifact_field', lambda result: getattr(result, 'custom_artifact_field', None))

    result = AttackResult(
        name='DLG',
        mse=0.1,
        iterations=1,
        time_seconds=0.0,
        success=False,
        success_threshold=0.5,
        gradient_mse=0.2,
    )
    result.custom_json_field = 7.0
    result.custom_artifact_field = 'hello'

    records = save_attack_artifacts(tmp_path, [result])
    payload = torch.load(tmp_path / records[0]['artifact_path'], map_location='cpu', weights_only=False)

    assert records[0]['custom_json_field'] == 7.0
    assert payload['custom_artifact_field'] == 'hello'


def test_attack_artifacts_persist_reconstructions(tmp_path):
    config = _tiny_patchtst_config()
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

    result = dlg_attack(config, state, target_update, x, y, device, target_type="update_payload", reference_inputs=x, reference_targets=y)
    result.client_id = "Nd2O3"
    result.round_index = 2
    result.sample_index = 0

    records = save_attack_artifacts(tmp_path, [result])

    assert len(records) == 1
    assert records[0]["artifact_path"] is not None
    artifact_path = tmp_path / records[0]["artifact_path"]
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert "real_x" not in payload
    assert "real_y" not in payload
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

    config = _tiny_patchtst_config()
    config['attack']['recovery_match_metric'] = 'l1'
    config['attack']['recovery_success_metric'] = 'l1'
    results = [
        AttackResult(
            name='DLG',
            mse=0.0,
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
    ts_config = _tiny_patchtst_config()
    ts_config["attack"]["success_rate_threshold"] = 0.5
    ts_config["attack"]["recovery_success_threshold"] = 0.01
    ts_results = [
        AttackResult(
            name="DLG",
            mse=0.0,
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
    assert ts_results[1].success is True
    assert ts_results[0].success_threshold == pytest.approx(0.5)

    cls_config = _tiny_classification_config()
    cls_config["attack"]["success_rate_threshold"] = 0.5
    cls_config["attack"]["recovery_success_threshold"] = 0.01
    cls_results = [
        AttackResult(
            name="iDLG",
            mse=0.0,
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
    assert cls_results[0].success is True
    assert cls_results[0].success_threshold == pytest.approx(0.5)
    assert cls_results[0].reference_y is not None
    assert torch.equal(cls_results[0].reference_y, torch.tensor([1]))





def test_classification_idlg_inferrs_label_from_update_payload():
    config = _tiny_classification_config()
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



def test_classification_update_payload_idlg_broadcasts_one_inferred_pseudo_label_for_multi_sample_batches():
    config = _tiny_classification_config()
    device = torch.device("cpu")
    model = build_model(config).to(device)
    x = torch.randn(2, 1, 4, 4)
    true_y = torch.tensor([1, 1], dtype=torch.long)
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
    assert result.reconstructed_y.shape == (2,)
    assert result.reconstructed_y.dtype == torch.long
    assert torch.equal(result.reconstructed_y.cpu(), true_y)
    assert not torch.equal(result.reconstructed_y.cpu(), decoy_y)



def test_time_series_idlg_degenerates_to_dlg_for_update_payloads():
    config = _tiny_patchtst_config()
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

    dlg = dlg_attack(config, state, target_update, x, decoy_y, device, target_type="update_payload")
    idlg = idlg_attack(config, state, target_update, x, decoy_y, device, target_type="update_payload")

    assert dlg.reconstructed_x is not None and idlg.reconstructed_x is not None
    assert dlg.reconstructed_y is not None and idlg.reconstructed_y is not None
    assert torch.allclose(idlg.reconstructed_x, dlg.reconstructed_x)
    assert torch.allclose(idlg.reconstructed_y, dlg.reconstructed_y)
    assert idlg.reconstructed_y.shape == true_y.shape
    assert torch.allclose(idlg.reconstructed_y.cpu(), dlg.reconstructed_y.cpu())
    assert idlg.gradient_mse == pytest.approx(dlg.gradient_mse)



def test_classification_update_payload_attacks_keep_reference_labels():
    config = _tiny_classification_config()
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

    assert dlg.metric_name == "objective_mse"
    assert dlg.reference_label == "nearest_client_train"
    assert dlg.reference_x.shape == x.shape
    assert dlg.nearest_client_train_indices is not None
    expected_reference_y = reference_targets.index_select(0, torch.tensor(dlg.nearest_client_train_indices, dtype=torch.long))
    assert dlg.reference_y is not None
    assert torch.equal(dlg.reference_y, expected_reference_y)
    assert dlg.reconstructed_y is not None
