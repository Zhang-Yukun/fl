import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from fedlab.federated.algorithms import load_captured_update_records, run_federated
from fedlab.utils.config import load_config


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
        "replay_capture.max_samples=1",
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
    sample = captures[0]["samples"][0]
    assert "real_x" not in sample
    assert "real_y" not in sample
    assert sample["sample_x_shape"][0] == 1
    assert sample["sample_x_dtype"] == "float32"

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
    assert "saved_updates" in common_content
    assert "attack_summary.json" in common_content
