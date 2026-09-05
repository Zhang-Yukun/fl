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
            "primary_metric_name": "budget_recovered_fraction",
            "reference_x": torch.tensor([[[1.0], [2.0], [3.0]], [[10.0], [20.0], [30.0]]]),
            "reference_y": torch.tensor([[[4.0], [5.0]], [[8.0], [9.0]]]),
            "reference_label": "nearest_client_train",
            "reconstructed_x": torch.tensor([[[1.1], [1.9], [3.2]], [[40.0], [50.0], [60.0]]]),
            "reconstructed_y": torch.tensor([[[4.2], [4.8]], [[7.0], [7.0]]]),
            "matched_reference_metric_name": "mse",
            "matched_reference_indices": [5, 8],
        },
        artifact_path,
    )
    records = [
        {
            "name": "DLG",
            "client_id": "Nd2O3",
            "round_index": 0,
            "sample_index": 0,
            "artifact_path": "attack_artifacts/round_0000/Nd2O3/sample_0000/dlg_00000.pt",
            "primary_metric_value": 0.12,
            "primary_metric_name": "budget_recovered_fraction",
        }
    ]
    (run_dir / "attack_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = module.select_candidates(run_dir, records, sort_key="matched_metric_value", descending=False, attack_name=None, client_id=None, round_index=None, limit=5)
    assert len(selected) == 2
    assert selected[0]["pair_index"] == 0
    assert selected[0]["matched_reference_index"] == 5
    assert selected[0]["matched_metric_value"] < selected[1]["matched_metric_value"]
    figure = module._attack_reconstruction_figure(selected[0]['result'])
    assert figure is not None
    assert figure._suptitle is not None
    assert 'loss_norm=0.020000' in figure._suptitle.get_text()
    assert 'loss_raw=0.020000' in figure._suptitle.get_text()

    output_dir = run_dir / "plots"
    plotted = [module.plot_candidate(candidate, output_dir) for candidate in selected]
    assert plotted[0].exists()
    report = module.build_report(selected, plotted)
    assert report[0] == "plotted 2 attack reconstructions"
    assert "pair=0" in report[1]


def test_plot_attack_reconstructions_script_exists():
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env python3")
    assert "attack_results.json" in content
    assert "artifact_path" in content
    assert "matched_metric_value" in content


def test_plot_attack_reconstructions_hides_idlg_target_by_default():
    record = {"name": "iDLG"}
    assert module.should_plot_real_y(record) is False
    assert module.should_plot_reconstructed_y(record) is False
    assert module.should_plot_real_y(record, show_policy_overrides=True) is True
    assert module.should_plot_reconstructed_y(record, show_policy_overrides=True) is True
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
            "primary_metric_name": "budget_recovered_fraction",
            "reference_x": torch.rand(2, 1, 4, 4),
            "reference_y": torch.tensor([2, 1]),
            "reference_label": "nearest_client_train",
            "reconstructed_x": torch.rand(2, 1, 4, 4),
            "reconstructed_y": torch.tensor([[0.1, 3.0, -1.0], [2.0, 0.5, -1.0]]),
            "matched_reference_metric_name": "mse",
            "matched_reference_indices": [2, 4],
        },
        artifact_path,
    )
    record = {
        "name": "DLG",
        "client_id": "client1",
        "round_index": 1,
        "sample_index": 0,
        "artifact_path": "attack_artifacts/round_0001/client1/sample_0000/dlg_00000.pt",
        "primary_metric_value": 0.08,
        "primary_metric_name": "budget_recovered_fraction",
    }

    candidates = module.expand_record_candidates(run_dir, record)
    assert len(candidates) == 2
    assert candidates[0]['result'].reference_y.shape == (1,)
    assert candidates[0]['result'].reconstructed_y.shape == (1, 3)

    output_path = module.plot_one_artifact(run_dir, record, run_dir / "plots")
    assert output_path.exists()


def test_plot_attack_reconstructions_supports_unknown_attack_policy_defaults():
    record = {"name": "CustomAttack"}
    assert module.should_plot_real_y(record) is True
    assert module.should_plot_reconstructed_y(record) is True
