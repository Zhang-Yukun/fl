import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch

import fedlab.federated.methods.encoded as encoded_methods
from fedlab.replay_capture.artifacts import load_captured_update_records
from fedlab.federated.algorithms import run_federated
from fedlab.utils.config import load_config




def _write_split(root: Path, client_id: str, split: str, images: torch.Tensor, labels: torch.Tensor) -> None:
    client_dir = root / 'clients' / client_id
    client_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': images, 'labels': labels}, client_dir / f'{split}.pt')


def _prepare_classification_split_dir(
    root: Path,
    *,
    client_ids: list[str],
    image_shape: tuple[int, int, int],
    num_classes: int,
) -> None:
    train_images_all = []
    train_labels_all = []
    val_images_all = []
    val_labels_all = []
    test_images_all = []
    test_labels_all = []
    for client_offset, client_id in enumerate(client_ids):
        base_value = float(client_offset) / 10.0
        train_images = torch.full((6, *image_shape), base_value, dtype=torch.float32)
        train_labels = torch.tensor([index % num_classes for index in range(6)], dtype=torch.long)
        val_images = torch.full((3, *image_shape), base_value + 0.1, dtype=torch.float32)
        val_labels = torch.tensor([index % num_classes for index in range(3)], dtype=torch.long)
        test_images = torch.full((3, *image_shape), base_value + 0.2, dtype=torch.float32)
        test_labels = torch.tensor([index % num_classes for index in range(3)], dtype=torch.long)
        _write_split(root, client_id, 'train', train_images, train_labels)
        _write_split(root, client_id, 'val', val_images, val_labels)
        _write_split(root, client_id, 'test', test_images, test_labels)
        train_images_all.append(train_images)
        train_labels_all.append(train_labels)
        val_images_all.append(val_images)
        val_labels_all.append(val_labels)
        test_images_all.append(test_images)
        test_labels_all.append(test_labels)
    server_dir = root / 'server'
    server_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': torch.cat(train_images_all, dim=0), 'labels': torch.cat(train_labels_all, dim=0)}, server_dir / 'train.pt')
    torch.save({'images': torch.cat(val_images_all, dim=0), 'labels': torch.cat(val_labels_all, dim=0)}, server_dir / 'val.pt')
    torch.save({'images': torch.cat(test_images_all, dim=0), 'labels': torch.cat(test_labels_all, dim=0)}, server_dir / 'test.pt')
    (root / 'summary.json').write_text(
        json.dumps({'class_names': [f'class_{index}' for index in range(num_classes)]}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


class _IdentityEgaCodec(torch.nn.Module):
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = int(block_size)
        self.encoded_dim = int(block_size)
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        return blocks.to(torch.float32)

    def decode_blocks(self, encoded_blocks: torch.Tensor) -> torch.Tensor:
        return encoded_blocks.to(torch.float32)


def _patch_identity_ega_codec(monkeypatch, *, block_size: int = 8) -> None:
    def _fake_load_ega_codec(config, device, num_clients, allow_pretrain):
        del config, device, num_clients, allow_pretrain
        return _IdentityEgaCodec(block_size=block_size)

    def _fake_load_ega_codec_payload(config, payload, device, num_clients):
        del config, payload, device, num_clients
        return _IdentityEgaCodec(block_size=block_size)

    monkeypatch.setattr(encoded_methods, 'load_ega_codec', _fake_load_ega_codec)
    monkeypatch.setattr(encoded_methods, 'load_ega_codec_payload', _fake_load_ega_codec_payload)


def _classification_config(
    split_dir: Path,
    output_dir: Path,
    *,
    client_ids: list[str],
    image_shape: tuple[int, int, int],
    num_classes: int,
) -> dict:
    return {
        'experiment': {'output_dir': str(output_dir), 'mode': 'federated'},
        'runtime': {'device': 'cpu', 'log_level': 'INFO', 'deterministic': True, 'seed': 2026},
        'task': {'type': 'classification'},
        'data': {
            'split_dir': str(split_dir),
            'clients': client_ids,
            'batch_size': 2,
            'shuffle_train': False,
            'num_workers': 0,
            'image_shape': list(image_shape),
            'num_classes': num_classes,
        },
        'model': {'name': 'small_cnn', 'hidden_channels': 4, 'dropout': 0.0},
        'training': {'epochs': 1, 'lr': 0.001, 'optimizer': 'adam', 'loss': 'cross_entropy', 'patience': 1, 'min_delta': 0.0},
        'centralized': {},
        'federated': {'algorithm': 'fedavg', 'rounds': 1},
        'replay_capture': {'enabled': True, 'frequency_rounds': 1},
        'attack': {'enabled': False, 'target_type': 'update_payload', 'steps': 1, 'async_enabled': False, 'device': 'same', 'max_samples': 1},
        'tracking': {'enabled': False},
        'evaluation': {'metrics': ['accuracy']},
        'artifacts': {'config_formats': ['yaml'], 'save_every_rounds': 0},
    }

TOOLS_DIR = Path(__file__).parents[2] / "fedlab" / "tools"
SCRIPT_PATH = TOOLS_DIR / "replay_saved_update_attacks.py"
DLG_SCRIPT_PATH = TOOLS_DIR / "replay_saved_update_dlg.py"
IDLG_SCRIPT_PATH = TOOLS_DIR / "replay_saved_update_idlg.py"
COMMON_SCRIPT_PATH = TOOLS_DIR / "replay_saved_update_common.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = _load_module(SCRIPT_PATH, "replay_saved_update_attacks")
dlg_module = _load_module(DLG_SCRIPT_PATH, "replay_saved_update_dlg")
idlg_module = _load_module(IDLG_SCRIPT_PATH, "replay_saved_update_idlg")


def _deterministic_overrides(output_dir: Path) -> list[str]:
    return [
        f"experiment.output_dir={output_dir}",
        "attack.target_type=update_payload",
        "replay_capture.frequency_rounds=1",
        "attack.max_samples=1",
        "attack.clients_per_round=1",
        "attack.client_selection=first",
        "attack.steps=1",
        "attack.optimizer=adam",
        "attack.local_optimizer=adam",
        "attack.async_enabled=false",
        "attack.seed=2026",
        "tracking.enabled=false",
        "runtime.device=cpu",
        "runtime.seed=2026",
        "runtime.deterministic=true",
        "data.shuffle_train=false",
        "model.dropout=0.0",
        "federated.algorithm=fedavg",
        "federated.rounds=1",
        "training.patience=1",
    ]


def _run_script(entry_module, script_name: str, source_dir: Path, replay_dir: Path) -> dict[str, object]:
    argv = [script_name, str(source_dir), "--output-dir", str(replay_dir)]
    stdout = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = argv
        with redirect_stdout(stdout):
            entry_module.main()
    finally:
        sys.argv = old_argv
    return json.loads(stdout.getvalue())


def test_replay_saved_update_attacks_runs_from_saved_updates_only(tmp_path):
    base_config = Path(__file__).parents[2] / "configs" / "test.yaml"
    source_dir = tmp_path / "source"
    replay_dir = tmp_path / "replay"

    source_config = load_config(base_config, _deterministic_overrides(source_dir))

    run_federated(source_config)

    captures = load_captured_update_records(source_dir)
    assert len(captures) == 3
    assert {record["client_id"] for record in captures} == {"Nd2O3", "CeO2", "La2O3"}
    assert (source_dir / "saved_updates" / "index.json").exists()
    assert "reference_inputs" not in captures[0]
    assert "reference_targets" not in captures[0]

    payload = _run_script(module, "replay_saved_update_attacks", source_dir, replay_dir)
    assert payload["attack_count"] == 2
    assert payload["summary_path"] == str(replay_dir / "summary.json")

    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    replay_results = json.loads((replay_dir / "attack_results.json").read_text(encoding="utf-8"))
    replay_summary = json.loads((replay_dir / "attack_summary.json").read_text(encoding="utf-8"))
    replay_run_summary = json.loads((replay_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(replay_results) == 2
    assert {record["name"] for record in replay_results} == {"DLG", "iDLG"}
    assert {record["target_type"] for record in replay_results} == {"update_payload"}
    assert replay_summary["primary_metric_name"] == "budget_recovered_fraction"
    assert replay_run_summary["test"] == source_summary["test"]
    assert replay_run_summary["rounds"] == source_summary["rounds"]
    assert replay_run_summary["attack_primary_metric_name"] == replay_summary["primary_metric_name"]
    assert replay_run_summary["attack_primary_metric_direction"] == replay_summary["primary_metric_direction"]
    assert replay_run_summary["attack_overall_avg_primary_metric_value"] == pytest.approx(
        replay_summary["overall_avg_primary_metric_value"]
    )
    assert replay_run_summary["attack_overall_best_primary_metric_value"] == pytest.approx(
        replay_summary["overall_best_primary_metric_value"]
    )
    assert replay_run_summary["attack_success_rate"] == pytest.approx(replay_summary["overall_success_rate"])
    assert replay_run_summary["attack_evaluations"] == len(replay_results)
    assert replay_run_summary["attack_summary"] == replay_summary
    assert sorted((replay_dir / "attack_artifacts").rglob("*.pt"))


def test_dedicated_replay_scripts_filter_methods(tmp_path):
    base_config = Path(__file__).parents[2] / "configs" / "test.yaml"
    source_dir = tmp_path / "source"
    dlg_dir = tmp_path / "dlg"
    idlg_dir = tmp_path / "idlg"

    source_config = load_config(base_config, _deterministic_overrides(source_dir))
    run_federated(source_config)

    dlg_payload = _run_script(dlg_module, "replay_saved_update_dlg", source_dir, dlg_dir)
    idlg_payload = _run_script(idlg_module, "replay_saved_update_idlg", source_dir, idlg_dir)

    assert dlg_payload["attack_count"] == 1
    assert idlg_payload["attack_count"] == 1

    dlg_results = json.loads((dlg_dir / "attack_results.json").read_text(encoding="utf-8"))
    idlg_results = json.loads((idlg_dir / "attack_results.json").read_text(encoding="utf-8"))

    assert [record["name"] for record in dlg_results] == ["DLG"]
    assert [record["name"] for record in idlg_results] == ["iDLG"]


def test_replay_saved_update_scripts_exist():
    for path in (SCRIPT_PATH, DLG_SCRIPT_PATH, IDLG_SCRIPT_PATH, COMMON_SCRIPT_PATH):
        assert path.exists()

    wrapper_content = SCRIPT_PATH.read_text(encoding="utf-8")
    common_content = COMMON_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "run_replay_cli" in wrapper_content
    assert "fedlab.attack.replay" in common_content
    assert "replay_saved_update_attacks" in common_content


def test_replay_saved_update_attacks_loads_classification_reference_data(tmp_path, monkeypatch):
    split_dir = tmp_path / 'split'
    source_dir = tmp_path / 'source_cls'
    replay_dir = tmp_path / 'replay_cls'
    _prepare_classification_split_dir(split_dir, client_ids=['m1', 'm2', 'm3'], image_shape=(1, 4, 4), num_classes=3)
    _patch_identity_ega_codec(monkeypatch)
    source_config = _classification_config(split_dir, source_dir, client_ids=['m1', 'm2', 'm3'], image_shape=(1, 4, 4), num_classes=3)

    run_federated(source_config)

    captures = load_captured_update_records(source_dir)
    assert captures
    assert 'reference_inputs' not in captures[0]
    assert 'reference_targets' not in captures[0]

    payload = _run_script(module, 'replay_saved_update_attacks', source_dir, replay_dir)
    assert payload['attack_count'] == 6
    replay_summary = json.loads((replay_dir / 'attack_summary.json').read_text(encoding='utf-8'))
    assert replay_summary['target_type'] == 'update_payload'
    assert replay_summary['primary_metric_name'] == 'budget_recovered_fraction'
