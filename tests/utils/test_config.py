from pathlib import Path

import pytest

from fedlab.federated.methods import FederatedMethod, MethodConfigSpec, register_method
from fedlab.utils.config import load_config


def test_nested_yaml_and_override():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.rounds=2", "tracking.enabled=false"])
    assert config["federated"]["rounds"] == 2
    assert config["data"]["seq_len"] == 21
    assert config["tracking"]["enabled"] is False





def test_nested_includes_deep_merge_dict_values(tmp_path):
    base_a = tmp_path / "a.yaml"
    base_b = tmp_path / "b.yaml"
    main = tmp_path / "main.yaml"

    base_a.write_text("""A:
  a: 1
""", encoding='utf-8')
    base_b.write_text("""A:
  b: 2
""", encoding='utf-8')
    main.write_text("""includes:
  - a.yaml
  - b.yaml
""", encoding='utf-8')

    config = load_config(main)

    assert config["A"] == {"a": 1, "b": 2}


def test_centralized_rounds_alias_maps_to_training_epochs():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["centralized.rounds=3", "tracking.enabled=false", "experiment.mode=centralized"])
    assert config["training"]["epochs"] == 3
    assert "centralized" not in config or "rounds" not in config.get("centralized", {})
    assert config["federated"]["rounds"] == 1


def test_load_config_prunes_irrelevant_algorithm_specific_fields():
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "federated.topk_fraction=0.25",
            "federated.quantization_dtype=int8",
            "adaptive_clipped_rdp.noise_multiplier=0.5",
            "ega.block_size=64",
            "privacy.clip_norm=2.0",
        ],
    )
    assert "topk_fraction" not in config["federated"]
    assert "quantization_dtype" not in config["federated"]
    assert "adaptive_clipped_rdp" not in config
    assert "ega" not in config
    assert "privacy" not in config


def test_load_config_keeps_only_active_algorithm_specific_fields():
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=secure_quantized_fedavg",
            "federated.topk_fraction=0.25",
            "federated.quantization_dtype=int8",
            "federated.quantization_seed=2026",
            "privacy.clip_norm=2.0",
            "privacy.noise_multiplier=0.1",
        ],
    )
    assert config["federated"]["quantization_dtype"] == "int8"
    assert config["federated"]["quantization_seed"] == 2026
    assert "topk_fraction" not in config["federated"]
    assert config["privacy"]["clip_norm"] == 2.0


def test_deprecated_centralized_epochs_key_is_rejected():
    with pytest.raises(ValueError, match=r"centralized\.epochs"):
        load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["centralized.epochs=3"])


def test_training_epochs_key_is_supported():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["training.epochs=3"])
    assert config["training"]["epochs"] == 3


def test_federated_local_epochs_alias_maps_to_training_epochs():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.local_epochs=4"])
    assert config["training"]["epochs"] == 4
    assert "local_epochs" not in config["federated"]


def test_conflicting_epoch_schedule_keys_are_rejected():
    with pytest.raises(ValueError, match=r"Conflicting epoch schedule keys"):
        load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["training.epochs=3", "federated.local_epochs=2"])


def test_load_config_materializes_runtime_defaults_for_saved_snapshots():
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        ["tracking.enabled=false"],
    )
    assert config["evaluation"]["mode"] == "protocol"
    assert config["evaluation"]["metrics"] == ["mse", "mae", "mape"]
    assert config["training"]["epochs"] == 1
    assert config["training"]["optimizer"] == "sgd"
    assert config["attack"]["model_mode"] == "train"
    assert config["attack"]["optimizer"] == "adam"
    assert config["attack"]["lr"] == 0.001
    assert config["attack"]["sample_count"] == "auto"
    assert config["attack"]["sample_count_cap"] == 8
    assert config["attack"]["max_samples"] == "auto"
    assert config["attack"]["max_samples_cap"] == 8
    assert config["attack"]["recovery_match_metric"] == "mse"
    assert config["attack"]["recovery_success_metric"] == "mse"
    assert config["grpc"]["max_message_mb"] == 256.0
    assert config["artifacts"]["config_formats"] == ["yaml"]


def test_load_config_uses_registered_method_config_metadata(monkeypatch):
    from fedlab.federated.methods import registry as method_registry

    snapshot = dict(method_registry._METHOD_REGISTRY)
    monkeypatch.setattr(method_registry, '_METHOD_REGISTRY', dict(snapshot))

    class CustomMetaMethod(FederatedMethod):
        name = 'custom_meta'
        config_spec = MethodConfigSpec(
            federated_keys=frozenset({'custom_fraction'}),
            root_blocks=frozenset({'custom_block'}),
            uses_privacy_block=True,
        )

        def client_update(self, **kwargs):
            return None

        def aggregate(self, **kwargs):
            return []

        def extract_attack_payload(self, **kwargs):
            return None

    register_method('custom_meta', CustomMetaMethod, compressed=False, description='custom test method')

    config = load_config(
        Path(__file__).parents[2] / 'configs' / 'test.yaml',
        [
            'federated.algorithm=custom_meta',
            'federated.custom_fraction=0.125',
            'federated.topk_fraction=0.25',
            'custom_block.enabled=true',
            'privacy.clip_norm=2.0',
        ],
    )

    assert config['federated']['custom_fraction'] == 0.125
    assert 'topk_fraction' not in config['federated']
    assert 'custom_block' in config
    assert config['privacy']['clip_norm'] == 2.0
