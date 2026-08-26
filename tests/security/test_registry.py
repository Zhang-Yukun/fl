from fedlab.security.registry import configured_attack_names, list_registered_attacks, register_attack, run_attacks


def test_builtin_attacks_are_registered():
    registered = list_registered_attacks()

    assert 'dlg' in registered
    assert 'idlg' in registered


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
