import torch

from fedlab.security.registry import (
    compute_recovery_metric_matrix,
    configured_attack_names,
    list_registered_attacks,
    list_registered_recovery_metrics,
    attack_primary_metric_direction,
    build_attack_artifact_payload,
    get_attack_summary_metric,
    list_registered_attack_artifact_fields,
    list_registered_attack_record_fields,
    list_registered_attack_summary_metrics,
    list_registered_attack_tracking_metrics,
    register_attack,
    register_attack_artifact_field,
    register_attack_record_field,
    register_attack_summary_metric,
    register_attack_tracking_metric,
    register_recovery_metric,
    resolve_recovery_objective,
    resolve_recovery_threshold,
    run_attacks,
    serialize_attack_record,
    summarize_metric_values,
)


def test_builtin_attacks_are_registered():
    registered = list_registered_attacks()

    assert 'dlg' in registered
    assert 'idlg' in registered


def test_builtin_attack_summary_metrics_are_registered():
    registered = list_registered_attack_summary_metrics()

    assert {'budget_recovered_fraction', 'coverage_recovered_fraction', 'objective_mse'} <= set(registered)
    assert 'reconstruction_mse' not in registered
    assert 'nearest_client_train_mse' not in registered
    assert attack_primary_metric_direction('budget_recovered_fraction') == 'lower_is_more_private'


def test_builtin_recovery_metrics_are_registered():
    registered = list_registered_recovery_metrics()

    assert {'mse', 'psnr', 'ssim'} <= set(registered)
    assert registered['mse'].default_objective == 'min'
    assert registered['psnr'].default_objective == 'max'


def test_configured_attack_names_default_to_dlg_and_idlg():
    assert configured_attack_names({'attack': {}}) == ('dlg', 'idlg')


def test_configured_attack_names_rejects_removed_method_key():
    import pytest

    with pytest.raises(ValueError, match=r"attack\.method"):
        configured_attack_names({'attack': {'method': 'dlg'}})


def test_configured_attack_names_supports_custom_selection(monkeypatch):
    import fedlab.security.registry as registry_module

    snapshot = list_registered_attacks()
    monkeypatch.setattr(registry_module, '_ATTACKS', dict(snapshot))
    monkeypatch.setattr(registry_module, '_ATTACK_ORDER', ['dlg', 'idlg'])
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACKS_LOADED', True)

    def custom_attack(*args, **kwargs):
        return {'name': 'custom'}

    register_attack('custom', custom_attack)

    assert configured_attack_names({'attack': {'methods': ['custom', 'dlg']}}) == ('custom', 'dlg')


def test_run_attacks_executes_registered_attacks_in_order(monkeypatch):
    import fedlab.security.registry as registry_module

    calls = []
    monkeypatch.setattr(registry_module, '_ATTACKS', {})
    monkeypatch.setattr(registry_module, '_ATTACK_ORDER', [])
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACKS_LOADED', True)

    def attack_a(*args, **kwargs):
        calls.append('a')
        return {'name': 'A'}

    def attack_b(*args, **kwargs):
        calls.append('b')
        return {'name': 'B'}

    register_attack('attack_a', attack_a)
    register_attack('attack_b', attack_b)

    results = run_attacks({'attack': {'methods': ['attack_b', 'attack_a']}}, None, None, None, None, None)

    assert calls == ['b', 'a']
    assert results == [{'name': 'B'}, {'name': 'A'}]


def test_register_recovery_metric_supports_custom_metric(monkeypatch):
    import fedlab.security.registry as registry_module

    snapshot = list_registered_recovery_metrics()
    monkeypatch.setattr(registry_module, '_RECOVERY_METRICS', dict(snapshot))
    monkeypatch.setattr(registry_module, '_BUILTIN_RECOVERY_METRICS_LOADED', True)

    def custom_matrix(reconstructed, reference_inputs, _data_range):
        return torch.zeros((reconstructed.shape[0], reference_inputs.shape[0]), dtype=torch.float32)

    register_recovery_metric(
        'custom_zero',
        custom_matrix,
        default_objective='min',
        default_threshold=lambda _config, _data_range: 0.25,
    )

    matrix = compute_recovery_metric_matrix(torch.zeros(2, 1), torch.zeros(3, 1), 'custom_zero', 1.0)

    assert matrix.shape == (2, 3)
    assert resolve_recovery_objective('auto', 'custom_zero') == 'min'
    assert resolve_recovery_threshold({'attack': {}}, 'custom_zero', 1.0) == 0.25


def test_register_attack_summary_metric_supports_custom_metric(monkeypatch):
    import fedlab.security.registry as registry_module

    snapshot = list_registered_attack_summary_metrics()
    monkeypatch.setattr(registry_module, '_ATTACK_SUMMARY_METRICS', dict(snapshot))
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACK_SUMMARY_METRICS_LOADED', True)

    register_attack_summary_metric(
        'custom_score',
        lambda result: getattr(result, 'custom_score', None),
        average_key='overall_avg_custom_score',
        best_key='overall_best_custom_score',
        best_objective='max',
        privacy_direction='lower_is_more_private',
    )

    class Dummy:
        custom_score = 0.75

    stats = summarize_metric_values([Dummy()], 'custom_score')

    assert get_attack_summary_metric('custom_score').best_objective == 'max'
    assert stats['average'] == 0.75
    assert stats['best'] == 0.75
    assert attack_primary_metric_direction('custom_score') == 'lower_is_more_private'


def test_attack_record_and_artifact_serialization_are_registry_driven(monkeypatch):
    import fedlab.security.registry as registry_module

    monkeypatch.setattr(registry_module, '_ATTACK_RECORD_FIELDS', dict(registry_module._ATTACK_RECORD_FIELDS))
    monkeypatch.setattr(registry_module, '_ATTACK_RECORD_FIELD_ORDER', list(registry_module._ATTACK_RECORD_FIELD_ORDER))
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACK_RECORD_FIELDS_LOADED', True)
    monkeypatch.setattr(registry_module, '_ATTACK_ARTIFACT_FIELDS', dict(registry_module._ATTACK_ARTIFACT_FIELDS))
    monkeypatch.setattr(registry_module, '_ATTACK_ARTIFACT_FIELD_ORDER', list(registry_module._ATTACK_ARTIFACT_FIELD_ORDER))
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACK_ARTIFACT_FIELDS_LOADED', True)

    register_attack_record_field('custom_json_field', lambda result: getattr(result, 'custom_json_field', None))
    register_attack_artifact_field('custom_artifact_field', lambda result: getattr(result, 'custom_artifact_field', None))

    class Dummy:
        custom_json_field = 1.5
        custom_artifact_field = 'payload'

    record = serialize_attack_record(Dummy())
    artifact = build_attack_artifact_payload(Dummy())

    assert 'custom_json_field' in record
    assert artifact['custom_artifact_field'] == 'payload'
    assert list_registered_attack_record_fields()
    assert list_registered_attack_artifact_fields()


def test_register_attack_tracking_metric_supports_custom_metric(monkeypatch):
    import fedlab.security.registry as registry_module

    snapshot = dict(registry_module._ATTACK_TRACKING_METRICS)
    order = list(registry_module._ATTACK_TRACKING_METRIC_ORDER)
    monkeypatch.setattr(registry_module, '_ATTACK_TRACKING_METRICS', dict(snapshot))
    monkeypatch.setattr(registry_module, '_ATTACK_TRACKING_METRIC_ORDER', list(order))
    monkeypatch.setattr(registry_module, '_BUILTIN_ATTACK_TRACKING_METRICS_LOADED', True)

    register_attack_tracking_metric('custom_track', lambda result: getattr(result, 'custom_track', None), current_key='custom_track')

    class Dummy:
        custom_track = 1.25

    tracked = {spec.name: spec for spec in list_registered_attack_tracking_metrics()}
    assert 'custom_track' in tracked
    assert tracked['custom_track'].value_getter(Dummy()) == 1.25
