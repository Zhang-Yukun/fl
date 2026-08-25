import importlib.util
import json
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[2] / "fedlab" / "tools" / "plot_attack_reconstructions.py"
spec = importlib.util.spec_from_file_location("plot_attack_reconstructions", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_plot_attack_reconstructions_plots_saved_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    artifacts = run_dir / "attack_artifacts" / "round_0000" / "Nd2O3" / "sample_0000"
    artifacts.mkdir(parents=True)
    artifact_path = artifacts / "dlg_00000.pt"
    torch.save(
        {
            "name": "DLG",
            "client_id": "Nd2O3",
            "round_index": 0,
            "sample_index": 0,
            "target_type": "update_payload",
            "primary_metric_name": "nearest_client_train_mse",
            "real_x": torch.tensor([[[9.0], [9.0], [9.0]]]),
            "real_y": torch.tensor([[[8.0], [8.0]]]),
            "reference_x": torch.tensor([[[1.0], [2.0], [3.0]]]),
            "reference_y": torch.tensor([[[4.0], [5.0]]]),
            "reference_label": "nearest_client_train",
            "reconstructed_x": torch.tensor([[[1.1], [1.9], [3.2]]]),
            "reconstructed_y": torch.tensor([[[4.2], [4.8]]]),
        },
        artifact_path,
    )
    (run_dir / "attack_results.json").write_text(
        json.dumps(
            [
                {
                    "name": "DLG",
                    "client_id": "Nd2O3",
                    "round_index": 0,
                    "sample_index": 0,
                    "artifact_path": "attack_artifacts/round_0000/Nd2O3/sample_0000/dlg_00000.pt",
                    "primary_metric_value": 0.12,
                    "primary_metric_name": "nearest_client_train_mse",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selected = module.select_records(module.load_json(run_dir / "attack_results.json"), sort_key="primary_metric_value", descending=False, attack_name=None, client_id=None, round_index=None, limit=5)
    assert len(selected) == 1

    output_dir = run_dir / "plots"
    output_path = module.plot_one_artifact(run_dir, selected[0], output_dir)
    assert output_path.exists()
    report = module.build_report(selected, [output_path])
    assert report[0] == "plotted 1 attack reconstructions"
    assert "Nd2O3" in report[1]


def test_plot_attack_reconstructions_script_exists():
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env python3")
    assert "attack_results.json" in content
    assert "artifact_path" in content


def test_plot_attack_reconstructions_hides_idlg_target_by_default():
    record = {"name": "iDLG"}
    assert module.should_plot_real_y(record) is False
    assert module.should_plot_reconstructed_y(record) is False
    assert module.should_plot_real_y(record, show_idlg_y=True) is True
    assert module.should_plot_reconstructed_y(record, show_idlg_y=True) is True
    assert module.should_plot_real_y({"name": "DLG"}) is True
    assert module.should_plot_reconstructed_y({"name": "DLG"}) is True


def test_plot_attack_reconstructions_supports_image_artifacts(tmp_path):
    run_dir = tmp_path / "run_image"
    artifacts = run_dir / "attack_artifacts" / "round_0001" / "client1" / "sample_0000"
    artifacts.mkdir(parents=True)
    artifact_path = artifacts / "dlg_00000.pt"
    torch.save(
        {
            "name": "DLG",
            "client_id": "client1",
            "round_index": 1,
            "sample_index": 0,
            "target_type": "update_payload",
            "primary_metric_name": "nearest_client_train_mse",
            "reference_x": torch.rand(1, 1, 4, 4),
            "reference_y": torch.tensor([2]),
            "reference_label": "nearest_client_train",
            "reconstructed_x": torch.rand(1, 1, 4, 4),
            "reconstructed_y": torch.randn(1, 3),
        },
        artifact_path,
    )
    (run_dir / "attack_results.json").write_text(
        json.dumps(
            [
                {
                    "name": "DLG",
                    "client_id": "client1",
                    "round_index": 1,
                    "sample_index": 0,
                    "artifact_path": "attack_artifacts/round_0001/client1/sample_0000/dlg_00000.pt",
                    "primary_metric_value": 0.08,
                    "primary_metric_name": "nearest_client_train_mse",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_path = module.plot_one_artifact(run_dir, module.load_json(run_dir / "attack_results.json")[0], run_dir / "plots")
    assert output_path.exists()
