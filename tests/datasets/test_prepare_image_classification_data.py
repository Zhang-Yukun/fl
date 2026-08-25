import importlib.util
import json
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[2] / "fedlab" / "tools" / "prepare_image_classification_data.py"
spec = importlib.util.spec_from_file_location("prepare_image_classification_data_script", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


build_client_split_payloads = module.build_client_split_payloads
save_client_split_payloads = module.save_client_split_payloads
reset_output_dir = module.reset_output_dir
_default_client_ids = module._default_client_ids


def test_build_client_split_payloads_is_deterministic_and_complete():
    train_images = torch.arange(24, dtype=torch.float32).reshape(6, 1, 2, 2) / 24.0
    train_labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    test_images = torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2) / 12.0
    test_labels = torch.tensor([2, 1, 0], dtype=torch.long)

    first = build_client_split_payloads(train_images, train_labels, test_images, test_labels, dataset_name='mnist', num_clients=3, val_ratio=0.5, seed=2026)
    second = build_client_split_payloads(train_images, train_labels, test_images, test_labels, dataset_name='mnist', num_clients=3, val_ratio=0.5, seed=2026)

    assert set(first) == {'m1', 'm2', 'm3'}
    for client_id in first:
        for split_name in ("train", "val", "test"):
            assert torch.equal(first[client_id][split_name]["images"], second[client_id][split_name]["images"])
            assert torch.equal(first[client_id][split_name]["labels"], second[client_id][split_name]["labels"])
    total_train = sum(payload["train"]["labels"].numel() + payload["val"]["labels"].numel() for payload in first.values())
    total_test = sum(payload["test"]["labels"].numel() for payload in first.values())
    assert total_train == train_labels.numel()
    assert total_test == test_labels.numel()


def test_save_client_split_payloads_writes_expected_layout(tmp_path):
    payloads = {
        "m1": {
            "train": {"images": torch.zeros(2, 1, 2, 2), "labels": torch.tensor([0, 1])},
            "val": {"images": torch.zeros(1, 1, 2, 2), "labels": torch.tensor([1])},
            "test": {"images": torch.zeros(1, 1, 2, 2), "labels": torch.tensor([0])},
        },
        "m2": {
            "train": {"images": torch.ones(2, 1, 2, 2), "labels": torch.tensor([2, 2])},
            "val": {"images": torch.ones(1, 1, 2, 2), "labels": torch.tensor([2])},
            "test": {"images": torch.ones(1, 1, 2, 2), "labels": torch.tensor([1])},
        },
        "m3": {
            "train": {"images": torch.full((2, 1, 2, 2), 2.0), "labels": torch.tensor([0, 0])},
            "val": {"images": torch.full((1, 1, 2, 2), 2.0), "labels": torch.tensor([1])},
            "test": {"images": torch.full((1, 1, 2, 2), 2.0), "labels": torch.tensor([2])},
        },
    }

    summary = save_client_split_payloads(
        tmp_path,
        "mnist",
        payloads,
        seed=2026,
        val_ratio=0.1,
        raw_root=tmp_path / "raw" / "mnist",
        class_names=[str(index) for index in range(3)],
    )

    assert (tmp_path / 'clients' / 'm1' / 'train.pt').exists()
    assert (tmp_path / 'clients' / 'm2' / 'val.pt').exists()
    assert (tmp_path / 'clients' / 'm3' / 'test.pt').exists()
    assert (tmp_path / 'server' / 'train.pt').exists()
    assert (tmp_path / 'server' / 'val.pt').exists()
    assert (tmp_path / 'server' / 'test.pt').exists()
    loaded = torch.load(tmp_path / 'clients' / 'm2' / 'train.pt', map_location="cpu", weights_only=False)
    assert torch.equal(loaded["labels"], torch.tensor([2, 2]))
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["dataset"] == "mnist"
    assert saved_summary['clients']['m1']['train']['rows'] == 2
    assert saved_summary['server']['train']['rows'] == 6
    assert summary["image_shape"] == [1, 2, 2]


def test_reset_output_dir_removes_generated_client_tree(tmp_path):
    client_dir = tmp_path / 'clients' / 'm1'
    client_dir.mkdir(parents=True)
    (client_dir / "train.pt").write_bytes(b"stale")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    reset_output_dir(tmp_path)

    assert not tmp_path.exists()



def test_default_client_ids_follow_dataset_prefixes():
    assert _default_client_ids('mnist', 3) == ['m1', 'm2', 'm3']
    assert _default_client_ids('cifar10', 3) == ['c1', 'c2', 'c3']
