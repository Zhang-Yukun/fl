import importlib.util
import json
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).parents[2] / "fedlab" / "tools" / "plot_attack_batch_suite.py"
spec = importlib.util.spec_from_file_location("plot_attack_batch_suite", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _write_run(run_dir: Path, algorithm: str, count_per_attack: int = 3) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps([{"round": 0, "algorithm": algorithm}], ensure_ascii=False), encoding="utf-8")
    records = []
    for attack_name in ("DLG", "iDLG"):
        for index in range(count_per_attack):
            rel = Path("attack_artifacts") / "round_0000" / "Nd2O3" / f"sample_{index:04d}" / f"{attack_name.lower()}_{index:05d}.pt"
            artifact_path = run_dir / rel
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "real_x": torch.tensor([[[1.0], [2.0], [3.0]]]),
                    "real_y": torch.tensor([[[4.0], [5.0]]]),
                    "reconstructed_x": torch.tensor([[[1.0 + index], [2.0], [3.0]]]),
                    "reconstructed_y": torch.tensor([[[4.0], [5.0]]]),
                },
                artifact_path,
            )
            records.append(
                {
                    "name": attack_name,
                    "client_id": "Nd2O3",
                    "round_index": 0,
                    "sample_index": index,
                    "artifact_path": rel.as_posix(),
                    "mse": float(index),
                    "budget_recovered_fraction": float(index) / max(count_per_attack, 1),
                    "primary_metric_name": "budget_recovered_fraction",
                }
            )
    (run_dir / "attack_results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def test_plot_attack_batch_suite_groups_by_algorithm_and_attack(tmp_path):
    fedavg = tmp_path / "fedavg_run"
    qint8 = tmp_path / "qint8_run"
    _write_run(fedavg, "fedavg", count_per_attack=4)
    _write_run(qint8, "secure_quantized_fedavg", count_per_attack=4)

    output_dir = tmp_path / "gallery"
    lines = module.plot_attack_suite([fedavg, qint8], output_dir, limit_per_attack=2)

    assert any("fedavg_run" in line for line in lines)
    assert any("qint8_run" in line for line in lines)
    fedavg_dlg = sorted((output_dir / "fedavg" / "DLG").glob("*.png"))
    fedavg_idlg = sorted((output_dir / "fedavg" / "iDLG").glob("*.png"))
    qint8_dlg = sorted((output_dir / "secure_quantized_fedavg" / "DLG").glob("*.png"))
    qint8_idlg = sorted((output_dir / "secure_quantized_fedavg" / "iDLG").glob("*.png"))
    assert len(fedavg_dlg) == 2
    assert len(fedavg_idlg) == 2
    assert len(qint8_dlg) == 2
    assert len(qint8_idlg) == 2
    assert (output_dir / "report.md").exists()


def test_plot_attack_batch_suite_script_exists():
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert content.startswith("#!/usr/bin/env python3")
    assert "limit_per_attack" in content
    assert "algorithm_label" in content


def test_plot_attack_batch_suite_discovers_custom_attack_names(tmp_path):
    run_dir = tmp_path / "custom_run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps([{"round": 0, "algorithm": "fedavg"}], ensure_ascii=False), encoding="utf-8")
    rel = Path("attack_artifacts") / "round_0000" / "Nd2O3" / "sample_0000" / "customattack_00000.pt"
    artifact_path = run_dir / rel
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "reference_x": torch.tensor([[[1.0], [2.0], [3.0]]]),
            "reference_y": torch.tensor([[[4.0], [5.0]]]),
            "reconstructed_x": torch.tensor([[[1.0], [2.1], [3.0]]]),
            "reconstructed_y": torch.tensor([[[4.0], [5.0]]]),
        },
        artifact_path,
    )
    (run_dir / "attack_results.json").write_text(
        json.dumps(
            [{
                "name": "CustomAttack",
                "client_id": "Nd2O3",
                "round_index": 0,
                "sample_index": 0,
                "artifact_path": rel.as_posix(),
                "primary_metric_value": 0.1,
                "primary_metric_name": "custom_privacy",
            }],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "gallery"
    module.plot_attack_suite([run_dir], output_dir, limit_per_attack=1)

    assert len(list((output_dir / "fedavg" / "CustomAttack").glob("*.png"))) == 1
