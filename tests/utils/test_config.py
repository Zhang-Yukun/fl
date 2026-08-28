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


def test_deprecated_centralized_rounds_key_is_rejected():
    with pytest.raises(ValueError, match=r"centralized\.rounds"):
        load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["centralized.rounds=3", "tracking.enabled=false", "experiment.mode=centralized"])


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


def test_deprecated_transport_mode_keys_are_rejected():
    with pytest.raises(ValueError, match=r"transport\.upload_mode"):
        load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["transport.upload_mode=update"])
    with pytest.raises(ValueError, match=r"transport\.download_mode"):
        load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["transport.download_mode=model"])


def test_training_epochs_key_is_supported():
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["training.epochs=3"])
    assert config["training"]["epochs"] == 3


def test_deprecated_federated_local_epochs_key_is_rejected():
    with pytest.raises(ValueError, match=r"federated\.local_epochs"):
        load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.local_epochs=4"])


def test_training_epochs_with_removed_legacy_epoch_key_is_rejected():
    with pytest.raises(ValueError, match=r"federated\.local_epochs"):
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
    assert config["grpc"]["max_message_mb"] == 384.0
    assert config["artifacts"]["config_formats"] == ["yaml"]




def test_config_tree_has_no_duplicate_yaml_keys_or_notebook_checkpoints():
    import yaml

    class _DupLoader(yaml.SafeLoader):
        pass

    def _construct_mapping(loader, node, deep=False):
        mapping = {}
        duplicates = []
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                duplicates.append(key)
            value = loader.construct_object(value_node, deep=deep)
            mapping[key] = value
        if duplicates:
            raise ValueError(f"duplicate keys {duplicates} at line {node.start_mark.line + 1}")
        return mapping

    _DupLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

    config_root = Path(__file__).parents[2] / "configs"
    assert not any(path.name == ".ipynb_checkpoints" for path in config_root.rglob("*"))
    for path in sorted(config_root.rglob("*.yaml")):
        yaml.load(path.read_text(encoding="utf-8"), Loader=_DupLoader)

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
