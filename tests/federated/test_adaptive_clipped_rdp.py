import json
from pathlib import Path

from fedlab.communication.grpc_training import GrpcFederatedCoordinator
from fedlab.federated.algorithms import run_federated
from fedlab.utils.config import load_config


CONFIG_DIR = Path(__file__).parents[2] / "configs"


def _base_overrides(tmp_path):
    return [
        "experiment.output_dir=" + str(tmp_path),
        "federated.algorithm=adaptive_clipped_rdp_fedavg",
        "federated.rounds=1",
        "training.patience=1",
        "tracking.enabled=false",
        "attack.enabled=false",
        "runtime.device=cpu",
        "runtime.seed=2026",
        "runtime.deterministic=true",
        "data.shuffle_train=false",
        "model.dropout=0.0",
        "adaptive_clipped_rdp.noise_multiplier=0.5",
        "adaptive_clipped_rdp.reference_clip_norm=1.0",
        "adaptive_clipped_rdp.min_clip_norm=0.1",
        "adaptive_clipped_rdp.max_clip_norm=2.0",
        "adaptive_clipped_rdp.clip_factor=1.0",
        "adaptive_clipped_rdp.rdp_alpha=8.0",
        "adaptive_clipped_rdp.delta=1e-5",
        "adaptive_clipped_rdp.total_clients=3",
        "adaptive_clipped_rdp.seed=2026",
    ]


def test_single_process_adaptive_clipped_rdp_records_privacy_summary(tmp_path):
    config = load_config(CONFIG_DIR / "test.yaml", _base_overrides(tmp_path))

    summary = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert summary["privacy_accountant"] == "adaptive_clipped_rdp"
    assert summary["privacy_trust_model"] == "central_dp_trusted_aggregator"
    assert summary["privacy_epsilon"] is not None
    assert summary["privacy_rdp_total"] is not None
    assert summary["adaptive_clip_norm"] is not None
    assert metrics[0]["algorithm"] == "adaptive_clipped_rdp_fedavg"
    assert metrics[0]["privacy_accountant"] == "adaptive_clipped_rdp"
    assert metrics[0]["adaptive_clip_norm"] is not None
    assert metrics[0]["privacy_rdp_round"] is not None
    assert all(client["aggregation_payload_kind"] == "dense_update" for client in metrics[0]["clients"])


def test_grpc_adaptive_clipped_rdp_records_privacy_summary(tmp_path):
    config = load_config(CONFIG_DIR / "test.yaml", _base_overrides(tmp_path))

    coordinator = GrpcFederatedCoordinator(config)
    global_payload = coordinator.get_global()
    assert global_payload["compressed"] is False

    from fedlab.datasets.rare_earth import build_federated_loaders
    from fedlab.federated.algorithms import resolve_device
    from fedlab.federated.client import FederatedClient

    train_loaders, _, _ = build_federated_loaders(config)
    device = resolve_device(config)
    response = None
    for client_id, loader in train_loaders.items():
        client = FederatedClient(client_id, loader, config, device)
        result = client.train(global_payload["state"], compressed=global_payload["compressed"], round_index=0)
        response = coordinator.submit_update({"round": 0, "result": result})
    assert response is not None and response["stop"] is True
    coordinator.finalize_if_requested()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert summary["privacy_accountant"] == "adaptive_clipped_rdp"
    assert summary["privacy_epsilon"] is not None
    assert metrics[0]["privacy_accountant"] == "adaptive_clipped_rdp"
