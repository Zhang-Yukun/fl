import torch

from fedlab.security.registry import (
    compute_recovery_metric_matrix,
    configured_attack_names,
    list_registered_attacks,
    list_registered_recovery_metrics,
    register_attack,
    register_recovery_metric,
    resolve_recovery_objective,
    resolve_recovery_threshold,
    run_attacks,
)


def test_builtin_attacks_are_registered():
    registered = list_registered_attacks()

    assert 'dlg' in registered
    assert 'idlg' in registered


def test_builtin_recovery_metrics_are_registered():
    registered = list_registered_recovery_metrics()

    assert {'mse', 'psnr', 'ssim'} <= set(registered)
    assert registered['mse'].default_objective == 'min'
    assert registered['psnr'].default_objective == 'max'


def test_configured_attack_names_default_to_dlg_and_idlg():
    assert configured_attack_names({'attack': {}}) == ('dlg', 'idlg')


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
