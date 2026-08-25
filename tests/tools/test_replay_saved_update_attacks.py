import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from fedlab.federated.algorithms import load_captured_update_records, run_federated
from fedlab.utils.config import load_config


SCRIPT_PATH = Path(__file__).parents[2] / "fedlab" / "tools" / "replay_saved_update_attacks.py"
spec = importlib.util.spec_from_file_location("replay_saved_update_attacks", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def _deterministic_overrides(output_dir: Path, attack_enabled: bool) -> list[str]:
    return [
        f"experiment.output_dir={output_dir}",
        f"attack.enabled={'true' if attack_enabled else 'false'}",
        "attack.target_type=update_payload",
        "attack.frequency_rounds=1",
        "attack.max_samples=1",
        "attack.sample_count=1",
        "attack.clients_per_round=1",
        "attack.client_selection=first",
        "attack.steps=1",
        "attack.optimizer=adam",
        "attack.local_optimizer=adam",
        "attack.async_enabled=false",
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


def test_replay_saved_update_attacks_matches_inline_results(tmp_path, monkeypatch):
    base_config = Path(__file__).parents[2] / "configs" / "test.yaml"
    online_dir = tmp_path / "online"
    source_dir = tmp_path / "source"
    replay_dir = tmp_path / "replay"

    online_config = load_config(base_config, _deterministic_overrides(online_dir, attack_enabled=True))
    source_config = load_config(base_config, _deterministic_overrides(source_dir, attack_enabled=False))

    run_federated(online_config)
    run_federated(source_config)

    captures = load_captured_update_records(source_dir)
    assert len(captures) == 3
    assert {record["client_id"] for record in captures} == {"Nd2O3", "CeO2", "La2O3"}
    assert (source_dir / "saved_updates" / "index.json").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        ["replay_saved_update_attacks", str(source_dir), "--output-dir", str(replay_dir)],
    )
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        module.main()
    payload = json.loads(stdout.getvalue())
    assert payload["attack_count"] == 2

    online_results = json.loads((online_dir / "attack_results.json").read_text(encoding="utf-8"))
    replay_results = json.loads((replay_dir / "attack_results.json").read_text(encoding="utf-8"))
    replay_summary = json.loads((replay_dir / "attack_summary.json").read_text(encoding="utf-8"))

    assert len(online_results) == len(replay_results) == 2
    for online, replay in zip(online_results, replay_results):
        assert online["name"] == replay["name"]
        assert online["client_id"] == replay["client_id"]
        assert online["round_index"] == replay["round_index"]
        assert online["sample_index"] == replay["sample_index"]
        assert online["target_type"] == replay["target_type"] == "update_payload"
        assert online["primary_metric_name"] == replay["primary_metric_name"]
        assert online["primary_metric_value"] == pytest.approx(replay["primary_metric_value"])
        assert online["nearest_client_train_mse"] == pytest.approx(replay["nearest_client_train_mse"])
        assert online["budget_recovered_fraction"] == pytest.approx(replay["budget_recovered_fraction"])
        assert online["objective_mse"] == pytest.approx(replay["objective_mse"])
        assert online["artifact_path"] == replay["artifact_path"]

    assert replay_summary["primary_metric_name"] == "budget_recovered_fraction"
    assert replay_summary["overall_avg_primary_metric_value"] == pytest.approx(
        sum(record["primary_metric_value"] for record in online_results) / len(online_results)
    )
    assert sorted((replay_dir / "attack_artifacts").rglob("*.pt"))


def test_replay_saved_update_attacks_script_exists():
    assert SCRIPT_PATH.exists()
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "saved_updates" in content
    assert "attack_summary.json" in content
