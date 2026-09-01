from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import fedlab.federated.algorithms as algorithms_module
import fedlab.attack.runner as runner_module
import fedlab.federated.client as client_module
import fedlab.federated.server as server_module
import fedlab.federated.methods.encoded as encoded_methods
from fedlab.datasets.rare_earth import build_federated_loaders
from fedlab.engine.training import train_n_steps
from fedlab.federated.client import FederatedClient, ClientResult
from fedlab.federated.protocol import validate_transport_modes
from fedlab.federated.methods.encoded import EGAFedAvgMethod
from fedlab.modeling.forecasting import build_model
from fedlab.utils.serialization import compress_topk, decompress_topk, dequantize_qsgd_state_update, dequantize_state_update, serialize_model
from fedlab.utils.transport import auxiliary_payload_num_bytes, auxiliary_payload_num_parameters, estimate_download_transport_bytes

from fedlab.attack.runner import resolve_attack_device
from fedlab.attack.tasks import build_update_attack_round_task
from fedlab.federated.algorithms import (
    _round_history_communication_summary,
    _wandb_cumulative_communication_payload,
    run_centralized,
    run_federated,
)
from fedlab.replay_capture.artifacts import load_captured_update_records
from fedlab.utils.config import load_config
from fedlab.utils.consistency import compare_fedavg_runs


def test_one_round_federated_run(tmp_path):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    assert result["rounds"] == 1
    assert 0.0 < result["last_parameter_download_compression_ratio"] < 1.0
    assert result["last_parameter_upload_compression_ratio"] == 1.0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics[0]["total_upload_bytes"] > 0
    assert metrics[0]["total_download_bytes"] > 0
    assert metrics[0]["total_parameter_upload_bytes"] == metrics[0]["total_upload_bytes"]
    assert metrics[0]["total_parameter_download_bytes"] == metrics[0]["total_download_bytes"]
    assert metrics[0]["total_transport_upload_bytes"] >= metrics[0]["total_parameter_upload_bytes"]
    assert metrics[0]["total_transport_download_bytes"] >= metrics[0]["total_parameter_download_bytes"]
    assert metrics[0]["model_parameters"] > 0
    assert {client["client_id"] for client in metrics[0]["clients"]} == {"Nd2O3", "CeO2", "La2O3"}
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "evaluation_context.json").exists()
    assert not (tmp_path / "config.json").exists()


def test_federated_run_saves_config_before_training_starts(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)

    def fail_build(_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(algorithms_module, "build_federated_loaders", fail_build)

    with pytest.raises(RuntimeError, match="boom"):
        run_federated(config)

    assert (tmp_path / "config.yaml").exists()


def test_centralized_run_saves_config_before_training_starts(tmp_path, monkeypatch):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml")
    config["experiment"]["output_dir"] = str(tmp_path)

    def fail_build(_config):
        raise RuntimeError("boom")

    monkeypatch.setattr(algorithms_module, "build_federated_loaders", fail_build)

    with pytest.raises(RuntimeError, match="boom"):
        run_centralized(config)

    assert (tmp_path / "config.yaml").exists()


def test_wandb_cumulative_communication_payload_uses_history_totals(tmp_path):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.algorithm=fedavg"])
    config["experiment"]["output_dir"] = str(tmp_path)
    run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    summary = _round_history_communication_summary([])

    assert summary["total_parameter_bytes"] == 0

    from fedlab.federated.server import RoundRecord

    record = RoundRecord(**metrics[0])
    payload = _wandb_cumulative_communication_payload([record])

    assert payload["cumulative/last_parameter_download_compression_ratio"] == metrics[0]["parameter_download_compression_ratio"]
    assert payload["cumulative/last_transport_download_compression_ratio"] == metrics[0]["transport_download_compression_ratio"]
    assert payload["cumulative/total_parameter_upload_bytes"] == metrics[0]["total_parameter_upload_bytes"]
    assert payload["cumulative/total_parameter_download_bytes"] == metrics[0]["total_parameter_download_bytes"]
    assert payload["cumulative/total_transport_bytes"] == metrics[0]["total_transport_bytes"]




def test_standard_fedavg_uses_dense_updates(tmp_path):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.algorithm=fedavg"])
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    assert result["last_parameter_upload_compression_ratio"] == 1.0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    round_metrics = metrics[0]
    context_download_bytes = sum(
        client["parameter_download_bytes"] - client["dense_download_reference_bytes"]
        for client in round_metrics["clients"]
    )
    assert all(client["aggregation_payload_kind"] == "dense_update" for client in round_metrics["clients"])
    assert round_metrics["total_upload_bytes"] == round_metrics["fedavg_reference_upload_bytes"]
    assert round_metrics["total_parameter_upload_bytes"] == round_metrics["fedavg_reference_upload_bytes"]
    assert round_metrics["total_download_bytes"] == round_metrics["fedavg_reference_download_bytes"] + context_download_bytes
    assert round_metrics["total_parameter_download_bytes"] == round_metrics["fedavg_reference_download_bytes"] + context_download_bytes
    assert 0.0 < round_metrics["parameter_download_compression_ratio"] < 1.0
    assert round_metrics["parameter_upload_compression_ratio"] == 1.0
    assert 0.0 < round_metrics["transport_download_compression_ratio"] < 1.0
    assert 0.0 < round_metrics["transport_upload_compression_ratio"] < 1.0


def test_fedavg_dense_update_bytes_remain_exact_after_first_round(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "federated.rounds=2",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "runtime.deterministic=true",
            "data.shuffle_train=false",
            f"experiment.output_dir={tmp_path}",
        ],
    )

    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert result["last_parameter_upload_compression_ratio"] == 1.0
    assert len(metrics) == 2
    for round_metrics in metrics:
        context_download_bytes = sum(
            client["parameter_download_bytes"] - client["dense_download_reference_bytes"]
            for client in round_metrics["clients"]
        )
        assert round_metrics["total_parameter_download_bytes"] == round_metrics["fedavg_reference_download_bytes"] + context_download_bytes
        assert round_metrics["total_parameter_upload_bytes"] == round_metrics["fedavg_reference_upload_bytes"]
        assert 0.0 < round_metrics["parameter_download_compression_ratio"] < 1.0
        assert round_metrics["parameter_upload_compression_ratio"] == 1.0
        for client in round_metrics["clients"]:
            assert client["parameter_upload_bytes"] == client["dense_upload_reference_bytes"]


class _TransportToyModel(torch.nn.Module):
    """Small deterministic model with parameters and floating buffers for transport tests."""

    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)
        self.linear = torch.nn.Linear(2, 2, bias=True)
        self.norm = torch.nn.LayerNorm(2)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()
            for buffer in self.buffers():
                buffer.zero_()

    def forward(self, x):
        return self.norm(self.linear(self.bn(x)))


class _ToyLoader:
    """Minimal loader carrying a dataset length for client sample accounting."""

    def __init__(self, length: int = 4):
        self.dataset = [(torch.zeros(2), torch.zeros(2)) for _ in range(length)]

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)


def _clone_state_dict(state):
    return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in state.items())


def _assert_state_float_value(state, expected: float) -> None:
    for name, tensor in state.items():
        if name.endswith("num_batches_tracked"):
            continue
        if tensor.dtype.is_floating_point:
            assert torch.allclose(tensor, torch.full_like(tensor, expected)), name


def _assert_full_state_equal(left, right):
    assert set(left.keys()) == set(right.keys())
    for name in left.keys():
        assert torch.equal(left[name], right[name]), name



def _transport_test_config(tmp_path) -> dict:
    return load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            f"experiment.output_dir={tmp_path}",
            "federated.algorithm=fedavg",
            "federated.rounds=3",
            "training.epochs=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "runtime.deterministic=true",
            "data.shuffle_train=false",
            "training.optimizer=sgd",
            "training.lr=0.1",
            "training.patience=999",
        ],
    )


def _run_manual_transport_rounds(config: dict, monkeypatch):
    def _build_toy_model(_config):
        return _TransportToyModel()

    def _fake_train_one_epoch(model, loader, optimizer, device):
        del loader, optimizer, device
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
            for buffer in model.buffers():
                if buffer.dtype.is_floating_point:
                    buffer.add_(1.0)
        return 0.0

    monkeypatch.setattr(client_module, "build_model", _build_toy_model)
    monkeypatch.setattr(server_module, "build_model", _build_toy_model)
    monkeypatch.setattr(client_module, "train_one_epoch", _fake_train_one_epoch)

    train_loaders = {"c1": _ToyLoader(), "c2": _ToyLoader()}
    val_loader = _ToyLoader(length=1)
    test_loader = _ToyLoader(length=1)
    device = torch.device("cpu")
    server = server_module.FederatedServer(config, val_loader, test_loader, device)
    total_train_samples = sum(len(loader.dataset) for loader in train_loaders.values())
    clients = [
        FederatedClient(
            client_id,
            loader,
            config,
            device,
            total_train_samples=total_train_samples,
            total_clients=len(train_loaders),
            allow_ega_pretrain=False,
        )
        for client_id, loader in train_loaders.items()
    ]

    rounds = []
    for round_index in range(3):
        round_base_state = _clone_state_dict(server.global_state)
        round_context = server.build_round_context()
        prepared_states = []
        results = []
        for client in clients:
            prepared = client.prepare_round_state(round_base_state, round_index=round_index, round_context=round_context)
            prepared_states.append((client.client_id, _clone_state_dict(prepared.download_state)))
            result = client.train(
                round_base_state,
                round_index=round_index,
                round_context=round_context,
                prepared_state=prepared,
            )
            results.append(result)
        aggregation_weights = server.aggregate_dense(
            results,
            round_index=round_index,
            round_base_state=round_base_state,
            round_context=round_context,
        )
        rounds.append({
            "round_index": round_index,
            "download_states": prepared_states,
            "results": results,
            "aggregation_weights": aggregation_weights,
            "global_state": _clone_state_dict(server.global_state),
        })
    return rounds


def test_single_node_transport_payloads_follow_fixed_dense_update_model_semantics(tmp_path, monkeypatch):
    config = _transport_test_config(tmp_path / "fixed_transport")

    rounds = _run_manual_transport_rounds(config, monkeypatch)

    expected_download_values = [0.0, 1.0, 2.0]
    expected_upload_values = [1.0, 1.0, 1.0]

    for round_result, expected_download, expected_upload in zip(rounds, expected_download_values, expected_upload_values):
        for _, download_state in round_result["download_states"]:
            _assert_state_float_value(download_state, expected_download)
        for result in round_result["results"]:
            assert result.aggregation_payload_kind == "dense_update"
            assert result.parameter_download_bytes == result.download_bytes
            assert result.parameter_upload_bytes == result.upload_bytes
            assert result.transport_download_bytes >= result.download_bytes
            assert result.transport_upload_bytes >= result.upload_bytes
            assert result.transport_download_overhead_bytes == result.transport_download_bytes - result.parameter_download_bytes
            assert result.transport_upload_overhead_bytes == result.transport_upload_bytes - result.parameter_upload_bytes
            assert result.parameter_download_parameters == result.download_parameters
            assert result.parameter_upload_parameters == result.upload_parameters
            _assert_state_float_value(result.aggregation_state, expected_upload)
        _assert_state_float_value(round_result["global_state"], float(round_result["round_index"] + 1))
        assert round_result["aggregation_weights"] == [0.5, 0.5]

    final_state = rounds[-1]["global_state"]
    assert "bn.running_mean" in final_state
    assert "bn.running_var" in final_state
    assert "linear.weight" in final_state
    assert "norm.weight" in final_state
    _assert_state_float_value(final_state, 3.0)


def _assert_selected_values(state, expected: dict[str, torch.Tensor]) -> None:
    for name, expected_tensor in expected.items():
        actual = state[name].detach().cpu().to(torch.float32)
        assert torch.allclose(actual, expected_tensor.detach().cpu().to(torch.float32)), name


def _run_manual_sparse_rounds(config: dict, monkeypatch):
    def _build_toy_model(_config):
        return _TransportToyModel()

    def _fake_train_one_epoch(model, loader, optimizer, device):
        del loader, optimizer, device
        with torch.no_grad():
            model.bn.weight.add_(torch.tensor([1.0, 2.0]))
            model.bn.running_mean.add_(1.0)
            model.bn.running_var.add_(1.0)
        return 0.0

    monkeypatch.setattr(client_module, "build_model", _build_toy_model)
    monkeypatch.setattr(server_module, "build_model", _build_toy_model)
    monkeypatch.setattr(client_module, "train_one_epoch", _fake_train_one_epoch)

    train_loaders = {"c1": _ToyLoader(), "c2": _ToyLoader()}
    val_loader = _ToyLoader(length=1)
    test_loader = _ToyLoader(length=1)
    device = torch.device("cpu")
    server = server_module.FederatedServer(config, val_loader, test_loader, device)
    total_train_samples = sum(len(loader.dataset) for loader in train_loaders.values())
    clients = [
        FederatedClient(
            client_id,
            loader,
            config,
            device,
            total_train_samples=total_train_samples,
            total_clients=len(train_loaders),
            allow_ega_pretrain=False,
        )
        for client_id, loader in train_loaders.items()
    ]

    rounds = []
    for round_index in range(3):
        round_base_state = _clone_state_dict(server.global_state)
        round_context = server.build_round_context()
        prepared_states = []
        results = []
        for client in clients:
            prepared = client.prepare_round_state(round_base_state, round_index=round_index, round_context=round_context)
            prepared_states.append((client.client_id, _clone_state_dict(prepared.download_state)))
            result = client.train(
                round_base_state,
                round_index=round_index,
                round_context=round_context,
                prepared_state=prepared,
            )
            results.append(result)
        aggregation_weights = server.aggregate_sparse(
            results,
            round_index=round_index,
            round_base_state=round_base_state,
            round_context=round_context,
        )
        rounds.append({
            "round_index": round_index,
            "download_states": prepared_states,
            "results": results,
            "aggregation_weights": aggregation_weights,
            "global_state": _clone_state_dict(server.global_state),
        })
    return rounds


def test_sparse_fedavg_transport_semantics_match_expected_payloads(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            f"experiment.output_dir={tmp_path / 'fixed'}",
            "federated.algorithm=sparse_fedavg",
            "federated.rounds=3",
            "training.epochs=1",
            "federated.topk_fraction=0.15",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "data.shuffle_train=false",
            "training.optimizer=sgd",
            "training.lr=0.1",
        ],
    )

    rounds = _run_manual_sparse_rounds(config, monkeypatch)
    expected_download_bn_weight = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0])]
    expected_download_buffer = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])]

    for round_result, expected_bn, expected_buffer in zip(rounds, expected_download_bn_weight, expected_download_buffer):
        for _, download_state in round_result["download_states"]:
            _assert_selected_values(
                download_state,
                {
                    "bn.weight": expected_bn,
                    "bn.running_mean": expected_buffer,
                    "bn.running_var": expected_buffer,
                },
            )
        for result in round_result["results"]:
            dense_sparse = decompress_topk(result.sparse_update)
            assert result.aggregation_payload_kind == "sparse_update"
            assert result.sparse_update.nbytes + client_module.state_num_bytes(result.aggregation_state) < result.dense_bytes
            assert result.parameter_upload_bytes == result.sparse_update.algorithm_num_bytes + client_module.state_num_bytes(result.aggregation_state)
            assert result.parameter_upload_parameters == result.sparse_update.algorithm_num_parameters + client_module.state_num_parameters(result.aggregation_state)
            assert result.parameter_upload_bytes > result.sparse_update.nbytes + client_module.state_num_bytes(result.aggregation_state)
            assert torch.allclose(dense_sparse["bn.weight"], torch.tensor([1.0, 2.0]))
            zero_trainable_keys = ["bn.bias", "linear.weight", "linear.bias", "norm.weight", "norm.bias"]
            for key in zero_trainable_keys:
                assert torch.count_nonzero(dense_sparse[key]).item() == 0
            _assert_selected_values(
                result.aggregation_state,
                {
                    "bn.running_mean": torch.tensor([1.0, 1.0]),
                    "bn.running_var": torch.tensor([1.0, 1.0]),
                },
            )
        expected_round = float(round_result["round_index"] + 1)
        _assert_selected_values(
            round_result["global_state"],
            {
                "bn.weight": torch.tensor([expected_round, expected_round * 2.0]),
                "bn.running_mean": torch.tensor([expected_round, expected_round]),
                "bn.running_var": torch.tensor([expected_round, expected_round]),
                "bn.bias": torch.tensor([0.0, 0.0]),
            },
        )
        assert round_result["aggregation_weights"] == [0.5, 0.5]


def _run_manual_quantized_rounds(config: dict, monkeypatch):
    def _build_toy_model(_config):
        return _TransportToyModel()

    def _fake_train_one_epoch(model, loader, optimizer, device):
        del loader, optimizer, device
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
            for buffer in model.buffers():
                if buffer.dtype.is_floating_point:
                    buffer.add_(1.0)
        return 0.0

    monkeypatch.setattr(client_module, "build_model", _build_toy_model)
    monkeypatch.setattr(server_module, "build_model", _build_toy_model)
    monkeypatch.setattr(client_module, "train_one_epoch", _fake_train_one_epoch)

    train_loaders = {"c1": _ToyLoader(), "c2": _ToyLoader()}
    val_loader = _ToyLoader(length=1)
    test_loader = _ToyLoader(length=1)
    device = torch.device("cpu")
    server = server_module.FederatedServer(config, val_loader, test_loader, device)
    total_train_samples = sum(len(loader.dataset) for loader in train_loaders.values())
    clients = [
        FederatedClient(
            client_id,
            loader,
            config,
            device,
            total_train_samples=total_train_samples,
            total_clients=len(train_loaders),
            allow_ega_pretrain=False,
        )
        for client_id, loader in train_loaders.items()
    ]

    rounds = []
    for round_index in range(3):
        round_base_state = _clone_state_dict(server.global_state)
        round_context = server.build_round_context()
        prepared_states = []
        results = []
        for client in clients:
            prepared = client.prepare_round_state(round_base_state, round_index=round_index, round_context=round_context)
            prepared_states.append((client.client_id, _clone_state_dict(prepared.download_state)))
            result = client.train(
                round_base_state,
                round_index=round_index,
                round_context=round_context,
                prepared_state=prepared,
            )
            results.append(result)
        aggregation_weights = server.aggregate_dense(
            results,
            round_index=round_index,
            round_base_state=round_base_state,
            round_context=round_context,
        )
        rounds.append({
            "round_index": round_index,
            "download_states": prepared_states,
            "results": results,
            "aggregation_weights": aggregation_weights,
            "global_state": _clone_state_dict(server.global_state),
        })
    return rounds


def test_secure_quantized_fedavg_transport_semantics_match_expected_payloads(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            f"experiment.output_dir={tmp_path / 'fixed_quantized'}",
            "federated.algorithm=secure_quantized_fedavg",
            "federated.rounds=3",
            "training.epochs=1",
            "federated.quantization_dtype=float16",
            "privacy.clip_norm=0.0",
            "privacy.noise_multiplier=0.0",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "data.shuffle_train=false",
            "training.optimizer=sgd",
            "training.lr=0.1",
        ],
    )

    rounds = _run_manual_quantized_rounds(config, monkeypatch)
    expected_download_value = [0.0, 1.0, 2.0]

    for round_result, expected_value in zip(rounds, expected_download_value):
        for _, download_state in round_result["download_states"]:
            assert all(tensor.dtype == torch.float16 for tensor in download_state.values())
            _assert_selected_values(
                dequantize_state_update(download_state),
                {
                    "bn.weight": torch.tensor([expected_value, expected_value]),
                    "bn.running_mean": torch.tensor([expected_value, expected_value]),
                    "linear.weight": torch.full((2, 2), expected_value),
                },
            )
        for result in round_result["results"]:
            assert result.aggregation_payload_kind == "quantized_update"
            assert result.parameter_upload_bytes < result.dense_bytes
            assert result.parameter_download_bytes < result.dense_download_reference_bytes
            assert all(tensor.dtype == torch.float16 for tensor in result.aggregation_state.values())
            dequantized_update = dequantize_state_update(result.aggregation_state)
            _assert_selected_values(
                dequantized_update,
                {
                    "bn.weight": torch.tensor([1.0, 1.0]),
                    "bn.running_mean": torch.tensor([1.0, 1.0]),
                    "linear.weight": torch.ones(2, 2),
                },
            )
        expected_global = float(round_result["round_index"] + 1)
        _assert_selected_values(
            round_result["global_state"],
            {
                "bn.weight": torch.tensor([expected_global, expected_global]),
                "bn.running_mean": torch.tensor([expected_global, expected_global]),
                "linear.weight": torch.full((2, 2), expected_global),
            },
        )
        assert round_result["aggregation_weights"] == [0.5, 0.5]


def test_saved_config_contains_materialized_runtime_defaults(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.rounds=1",
            "attack.enabled=false",
            "tracking.enabled=false",
        ],
    )
    run_federated(config)
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["evaluation"]["mode"] == "protocol"
    assert saved["attack"]["model_mode"] == "train"
    assert saved["attack"]["recovery_success_threshold"] == 0.5


def test_config_artifact_formats_are_configurable(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        ["artifacts.config_formats=[yaml,json,toml]"],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    run_federated(config)
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "config.toml").exists()


def test_federated_run_saves_periodic_snapshot(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "federated.rounds=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "artifacts.save_every_rounds=1",
        ],
    )
    run_federated(config)

    snapshot_dir = tmp_path / "snapshots" / "round_0001"
    assert (snapshot_dir / "config.yaml").exists()
    assert (snapshot_dir / "metrics.json").exists()
    assert (snapshot_dir / "summary.json").exists()
    assert (snapshot_dir / "model.pt").exists()
    assert (snapshot_dir / "resume_state.pt").exists()


def test_federated_run_saves_update_captures_for_offline_replay(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "federated.rounds=1",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)

    result = run_federated(config)
    captures = load_captured_update_records(tmp_path)

    assert 'attack_evaluations' not in result
    assert len(captures) == 3
    assert {record['client_id'] for record in captures} == {'Nd2O3', 'CeO2', 'La2O3'}
    assert (tmp_path / 'saved_updates' / 'index.json').exists()
    assert 'reference_inputs' not in captures[0]
    assert 'reference_targets' not in captures[0]
    assert not (tmp_path / 'attack_results.json').exists()


def test_federated_run_ignores_online_attack_execution_flags(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "attack.enabled=true",
            "attack.methods=dlg",
            "attack.target_type=update_payload",
            "replay_capture.frequency_rounds=1",
            "attack.max_samples=1",
            "attack.steps=1",
            "federated.algorithm=fedavg",
            "federated.rounds=1",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)

    result = run_federated(config)

    assert 'attack_evaluations' not in result
    assert not (tmp_path / 'attack_results.json').exists()


def test_attack_max_samples_auto_defaults_to_capped_training_set_size():
    attack_cfg = {
        "max_samples": "auto",
        "max_samples_cap": 8,
    }

    assert runner_module._resolve_attack_max_samples({"attack": attack_cfg}, available_samples=18000) == 8
    assert runner_module._resolve_attack_max_samples({"attack": {"max_samples": "auto", "max_samples_cap": 32}}, available_samples=20) == 20
    assert runner_module._resolve_attack_max_samples({"attack": {"max_samples": 3}}, available_samples=18000) == 3



def test_attack_task_uses_uploaded_protocol_payload():
    config = {
        "federated": {"algorithm": "sparse_fedavg", "topk_fraction": 0.5},
        "replay_capture": {
            "enabled": True,
            "frequency_rounds": 1,
        },
        "attack": {
            "enabled": True,
            "target_type": "update_payload",
            "client_selection": "all",
            "max_samples": 1,
        },
    }
    protocol_update = serialize_model(torch.nn.Linear(2, 1, bias=False))
    protocol_update["weight"] = torch.tensor([[0.0, 2.0]])
    result = client_module.ClientResult(
        client_id="Nd2O3",
        num_samples=1,
        loss=0.0,
        sparse_update=compress_topk(protocol_update, 0.5),
        aggregation_payload_kind="sparse_update",
    )
    client = SimpleNamespace(
        client_id="Nd2O3",
        sample_batch=lambda max_samples=None, batch_index=0: (torch.zeros(1, 2, 1), torch.zeros(1, 1, 1)),
    )

    records = algorithms_module._capture_round_update_records(
        config,
        [client],
        [result],
        round_index=0,
        round_base_state=serialize_model(torch.nn.Linear(2, 1, bias=False)),
        server=None,
        round_context={},
        max_rounds=1,
    )
    task = build_update_attack_round_task(
        config,
        records,
        round_index=0,
        max_rounds=1,
    )

    assert task is not None
    assert task.samples[0].sample_index == 0
    target = task.samples[0].target
    assert isinstance(target, dict)
    assert torch.equal(target["weight"], torch.tensor([[0.0, 2.0]]))


def test_dense_attack_payload_uses_uploaded_update_directly():
    config = {"federated": {"algorithm": "fedavg"}}
    uploaded_update = OrderedDict([("weight", torch.tensor([[1.0, 3.0]]))])
    result = ClientResult(
        client_id="Nd2O3",
        num_samples=1,
        loss=0.0,
        aggregation_state=uploaded_update,
        aggregation_payload_kind="dense_update",
    )

    payload = algorithms_module._extract_attack_payload(
        config,
        result,
        [result],
        server=None,
        round_base_state=None,
        round_index=0,
        round_context={},
    )

    assert torch.equal(payload["weight"], uploaded_update["weight"])


def test_attack_payload_merges_sparse_and_dense_buffer_updates():
    config = {"federated": {"algorithm": "randomk_fedavg"}}
    protocol_update = OrderedDict([("weight", torch.tensor([[0.0, 2.0]]))])
    buffer_update = OrderedDict([("running_var", torch.tensor([0.25, 0.5]))])
    result = client_module.ClientResult(
        client_id="Nd2O3",
        num_samples=1,
        loss=0.0,
        aggregation_state=buffer_update,
        sparse_update=compress_topk(protocol_update, 1.0),
        aggregation_payload_kind="randomk_update",
    )

    payload = algorithms_module._extract_attack_payload(config, result, [result])

    assert set(payload.keys()) == {"weight", "running_var"}
    assert torch.equal(payload["weight"], protocol_update["weight"])
    assert torch.equal(payload["running_var"], buffer_update["running_var"])


def test_sparse_aggregation_merges_dense_buffer_updates(tmp_path):
    config = load_config(Path(__file__).parents[2] / "configs" / "test.yaml", ["federated.algorithm=randomk_fedavg"])
    config["experiment"]["output_dir"] = str(tmp_path)
    _, val_loader, test_loader = build_federated_loaders(config)
    server = server_module.FederatedServer(config, val_loader, test_loader, torch.device("cpu"))
    server.global_state = OrderedDict([
        ("weight", torch.zeros(2)),
        ("running_var", torch.ones(2)),
    ])
    round_base_state = OrderedDict((name, tensor.clone()) for name, tensor in server.global_state.items())
    results = [
        client_module.ClientResult(
            client_id="c1",
            num_samples=1,
            loss=0.0,
            aggregation_state=OrderedDict([("running_var", torch.tensor([0.5, -0.25]))]),
            sparse_update=compress_topk(OrderedDict([("weight", torch.tensor([1.0, 0.0]))]), 1.0),
            aggregation_payload_kind="randomk_update",
        ),
        client_module.ClientResult(
            client_id="c2",
            num_samples=3,
            loss=0.0,
            aggregation_state=OrderedDict([("running_var", torch.tensor([0.25, 0.75]))]),
            sparse_update=compress_topk(OrderedDict([("weight", torch.tensor([0.0, 2.0]))]), 1.0),
            aggregation_payload_kind="randomk_update",
        ),
    ]

    server.aggregate_sparse(results, round_base_state=round_base_state)

    assert torch.allclose(server.global_state["weight"], torch.tensor([0.25, 1.5]))
    assert torch.allclose(server.global_state["running_var"], torch.tensor([1.3125, 1.5]))


def test_randomk_fedavg_uses_unbiased_sparse_payloads(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=randomk_fedavg",
            "federated.rounds=1",
            "federated.topk_fraction=0.25",
            "federated.randomk_seed=2026",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]

    assert result["last_parameter_upload_compression_ratio"] > 1.0
    assert all(client["aggregation_payload_kind"] == "randomk_update" for client in clients)
    assert all(client["compressor"] == "randomk_unbiased" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)



def test_sign_fedavg_uses_sign_quantized_dense_updates(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=sign_fedavg",
            "federated.rounds=1",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]

    assert result["last_parameter_upload_compression_ratio"] > 2.0
    assert all(client["aggregation_payload_kind"] == "sign_update" for client in clients)
    assert all(client["compressor"] == "sign_mean_abs" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)


def test_secure_quantized_fedavg_uses_quantized_dense_updates(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=secure_quantized_fedavg",
            "federated.rounds=1",
            "federated.quantization_dtype=float16",
            "privacy.clip_norm=10.0",
            "privacy.noise_multiplier=0.0",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]

    assert result["last_parameter_upload_compression_ratio"] > 1.5
    assert result["last_parameter_total_communication_ratio"] > 1.9
    assert all(client["aggregation_payload_kind"] == "quantized_update" for client in clients)
    assert all(client["compressor"] == "float16_quantized_dense" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)
    assert all(client["parameter_download_bytes"] > 0 for client in clients)
    assert all(client["parameter_upload_bytes"] == client["upload_bytes"] for client in clients)
    assert all(client["parameter_download_bytes"] == client["download_bytes"] for client in clients)


def test_qsgd_fedavg_uses_stochastic_multilevel_quantization(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=qsgd_fedavg",
            "federated.rounds=1",
            "federated.qsgd_levels=31",
            "federated.quantization_seed=2026",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]

    assert result["last_parameter_upload_compression_ratio"] > 1.5
    assert all(client["aggregation_payload_kind"] == "qsgd_update" for client in clients)
    assert all(client["compressor"] == "qsgd_31_levels" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)


def test_ega_fedavg_uses_encoded_gradient_aggregation(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=ega_fedavg",
            "federated.rounds=1",
            "federated.quantization_seed=2026",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    config["ega"] = {
        "artifact_path": str(tmp_path / "ega_codec.pt"),
        "block_size": 8,
        "encoded_dim": 4,
        "hidden_dim": 16,
        "residual_blocks": 1,
        "quantization_level": 16,
        "normalization": 2e-4,
        "initial_normalization": 2e-4,
        "min_normalization": 1e-6,
        "normalization_strategy": "reported_client_max_abs",
        "encoded_dtype": "int8",
        "error_feedback": True,
        "pretrain": {
            "device": "cpu",
            "epochs": 2,
            "batch_size": 16,
            "lr": 1e-3,
            "train_groups": 64,
            "val_groups": 16,
            "seed": 2026,
        },
    }
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]

    assert (tmp_path / "ega_codec.pt").exists()
    assert result["last_parameter_upload_compression_ratio"] > 1.0
    assert result["best_val_mse"] == result["best_val_mse"]
    assert all(client["aggregation_payload_kind"] == "ega_encoded_update" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)
    assert all(client["parameter_download_bytes"] > 0 for client in clients)


def test_ega_protocol_aggregation_uses_client_visible_base(monkeypatch):
    method = EGAFedAvgMethod()
    server = SimpleNamespace(
        ega_codec=object(),
        ega_trainable_keys=("weight",),
        config={"ega": {}},
        global_state=OrderedDict([("weight", torch.tensor([10.0]))]),
        ega_normalization=1.0,
    )
    protocol_base = OrderedDict([("weight", torch.tensor([6.5]))])
    decoded_update = OrderedDict([("weight", torch.tensor([1.0]))])
    results = [
        client_module.ClientResult(client_id="c1", num_samples=1, loss=0.0, ega_payload=object()),
        client_module.ClientResult(client_id="c2", num_samples=3, loss=0.0, ega_payload=object()),
    ]

    monkeypatch.setattr(
        "fedlab.federated.methods.encoded.decode_mean_encoded_payload",
        lambda payloads, codec: decoded_update,
    )
    monkeypatch.setattr(
        method,
        "_protocol_base_state",
        lambda **kwargs: protocol_base,
    )

    method.aggregate(server=server, results=results, round_base_state=OrderedDict([("weight", torch.tensor([10.0]))]), round_index=3)

    assert torch.allclose(server.global_state["weight"], torch.tensor([7.5]))


def test_secure_quantized_fedavg_supports_absmax_int8(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=secure_quantized_fedavg",
            "federated.rounds=1",
            "federated.quantization_dtype=int8",
            "federated.quantization_stochastic_rounding=true",
            "federated.quantization_seed=2026",
            "privacy.clip_norm=10.0",
            "privacy.noise_multiplier=0.0",
            "attack.enabled=false",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    result = run_federated(config)
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    clients = metrics[0]["clients"]

    assert result["last_parameter_upload_compression_ratio"] > 3.0
    assert result["last_parameter_total_communication_ratio"] > 3.0
    assert all(client["compressor"] == "int8_quantized_dense" for client in clients)
    assert all(client["upload_bytes"] < client["dense_upload_reference_bytes"] for client in clients)
    assert all(client["parameter_download_bytes"] > 0 for client in clients)
    assert all(client["parameter_upload_bytes"] == client["upload_bytes"] for client in clients)
    assert all(client["parameter_download_bytes"] == client["download_bytes"] for client in clients)



def test_train_n_steps_cycles_loader_and_validates_steps():
    device = torch.device("cpu")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [(torch.ones(2, 1), torch.zeros(2, 1))]

    with pytest.raises(ValueError):
        train_n_steps(model, loader, optimizer, device, steps=0)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_n_steps(model, loader, optimizer, device, steps=3)
    after = list(model.parameters())

    assert loss >= 0.0
    assert any(not torch.allclose(left, right.detach()) for left, right in zip(before, after))


def test_client_prefers_local_steps_over_training_epochs(monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "federated.local_steps=2",
            "training.epochs=7",
            "attack.enabled=false",
        ],
    )
    config["runtime"]["device"] = "cpu"
    train_loaders, _, _ = build_federated_loaders(config)
    client_id, loader = next(iter(train_loaders.items()))
    client = FederatedClient(client_id, loader, config, torch.device("cpu"))
    global_state = serialize_model(build_model(config))
    calls = {"steps": 0, "epochs": 0}

    def fake_train_n_steps(model, loader, optimizer, device, steps):
        calls["steps"] += 1
        assert steps == 2
        return 1.25

    def fake_train_one_epoch(model, loader, optimizer, device):
        calls["epochs"] += 1
        return 9.0

    monkeypatch.setattr(client_module, "train_n_steps", fake_train_n_steps)
    monkeypatch.setattr(client_module, "train_one_epoch", fake_train_one_epoch)

    result = client.train(global_state, compressed=False, round_index=0)

    assert calls == {"steps": 1, "epochs": 0}
    assert result.loss == 1.25


class _PredictionTrackerStub:
    def __init__(self):
        self.logs = []
        self.prediction_logs = []

    def log(self, data, step=None):
        self.logs.append((step, data))

    def log_prediction_plot(self, key, input_series, prediction, target, step=None, title=None, scaler=None):
        self.prediction_logs.append((key, step, title, scaler))


def test_log_prediction_views_adds_client_specific_keys(monkeypatch):
    tracker = _PredictionTrackerStub()
    loader_a = SimpleNamespace(scaler="scale_a")
    loader_b = SimpleNamespace(scaler="scale_b")
    merged_loader = SimpleNamespace(loaders=[loader_a, loader_b], scaler="scale_merged")
    sample = (torch.zeros(1, 2, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))

    monkeypatch.setattr(algorithms_module, "_predict_for_logging", lambda model, loader, device, state=None: sample)

    algorithms_module._log_prediction_views(
        tracker,
        "prediction/federated/val_protocol",
        "federated val protocol prediction",
        model=None,
        loader=merged_loader,
        device=torch.device("cpu"),
        step=5,
        client_ids=["Nd2O3", "CeO2"],
        state=OrderedDict(),
    )

    keys = [item[0] for item in tracker.prediction_logs]
    assert "prediction/federated/val_protocol" in keys
    assert "prediction/federated/val_protocol/client/Nd2O3" in keys
    assert "prediction/federated/val_protocol/client/CeO2" in keys


class _DummyLoader:
    def __init__(self, size: int = 1):
        self.dataset = [(torch.zeros(1, 1), torch.zeros(1, 1)) for _ in range(size)]

    def __iter__(self):
        return iter(self.dataset)



def test_centralized_run_restores_best_validation_checkpoint(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "training.epochs=2",
            "training.patience=10",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    val_loader = object()
    test_loader = object()
    model = torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(
        algorithms_module,
        "build_federated_loaders",
        lambda _config: ({"client_a": _DummyLoader()}, val_loader, test_loader),
    )
    monkeypatch.setattr(algorithms_module, "build_model", lambda _config: model)

    round_state = {"count": 0}

    def fake_train_one_epoch(model, loader, optimizer, device):
        round_state["count"] += 1
        with torch.no_grad():
            model.weight.fill_(float(round_state["count"]))
        return float(round_state["count"])

    def fake_evaluate(model, loader, device):
        weight = float(model.weight.item())
        if loader is val_loader:
            mse = 0.1 if weight == 1.0 else 0.2
            return {"mse": mse, "mae": weight, "mape": weight}
        if loader is test_loader:
            mse = 10.0 if weight == 1.0 else 20.0
            return {"mse": mse, "mae": mse / 10.0, "mape": mse / 5.0}
        raise AssertionError("unexpected loader")

    monkeypatch.setattr(algorithms_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(algorithms_module, "evaluate", fake_evaluate)

    result = run_centralized(config)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    saved_state = torch.load(tmp_path / "model.pt", map_location="cpu")

    assert result["mse"] == 10.0
    assert summary["best_round"] == 0
    assert summary["best_val_mse"] == 0.1
    assert summary["test_checkpoint"] == "best_validation"
    assert metrics["best_round"] == 0
    assert metrics["best_val"]["mse"] == 0.1
    assert metrics["history"][0]["round"] == 0
    assert "epoch" not in metrics["history"][0]
    assert float(saved_state["weight"].item()) == pytest.approx(1.0)



def test_federated_run_restores_best_validation_checkpoint(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "federated.rounds=2",
            "training.patience=10",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    config["experiment"]["output_dir"] = str(tmp_path)
    val_loader = object()
    test_loader = object()

    monkeypatch.setattr(
        algorithms_module,
        "build_federated_loaders",
        lambda _config: (
            {"Nd2O3": _DummyLoader(), "CeO2": _DummyLoader(), "La2O3": _DummyLoader()},
            val_loader,
            test_loader,
        ),
    )
    def _build_zero_model(_config):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        return model

    monkeypatch.setattr(server_module, "build_model", _build_zero_model)

    def fake_client_train(self, global_state, compressed=False, round_index=0, round_context=None, prepared_state=None):
        update = type(global_state)((name, torch.ones_like(tensor)) for name, tensor in global_state.items())
        return client_module.ClientResult(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            loss=float(round_index + 1),
            aggregation_state=update,
            dense_bytes=4,
            dense_parameters=1,
            download_bytes=4,
            download_parameters=1,
            parameter_download_bytes=4,
            parameter_download_parameters=1,
            dense_download_reference_bytes=4,
            dense_download_reference_parameters=1,
            upload_bytes=4,
            upload_parameters=1,
            parameter_upload_bytes=4,
            parameter_upload_parameters=1,
            transport_download_bytes=4,
            transport_upload_bytes=4,
        )

    def fake_evaluate(model, loader, device):
        weight = float(model.weight.item())
        if loader is val_loader:
            mse = 0.1 if weight == 1.0 else 0.2
            return {"mse": mse, "mae": weight, "mape": weight}
        if loader is test_loader:
            mse = 10.0 if weight == 1.0 else 20.0
            return {"mse": mse, "mae": mse / 10.0, "mape": mse / 5.0}
        raise AssertionError("unexpected loader")

    monkeypatch.setattr(client_module.FederatedClient, "train", fake_client_train)
    monkeypatch.setattr(server_module, "evaluate", fake_evaluate)

    summary = run_federated(config)
    saved_state = torch.load(tmp_path / "model.pt", map_location="cpu")
    persisted_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert summary["test"]["mse"] == 10.0
    assert persisted_summary["best_round"] == 0
    assert persisted_summary["best_val_mse"] == 0.1
    assert persisted_summary["test_checkpoint"] == "best_validation"
    assert float(saved_state["weight"].item()) == pytest.approx(1.0)


def test_client_train_returns_aggregation_payload():
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "federated.algorithm=fedavg",
            "attack.enabled=false",
            "runtime.device=cpu",
        ],
    )
    train_loaders, _, _ = build_federated_loaders(config)
    client_id, loader = next(iter(train_loaders.items()))
    client = FederatedClient(client_id, loader, config, torch.device("cpu"))
    global_state = serialize_model(build_model(config))

    result = client.train(global_state, compressed=False, round_index=0)

    assert result.aggregation_state is not None



def test_attack_device_defaults_to_runtime_device():
    config = {"runtime": {"device": "cpu"}, "attack": {}}

    assert str(resolve_attack_device(config)) == "cpu"


def test_one_round_federated_run_with_sgd_optimizer(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "training.optimizer=sgd",
            "training.lr=0.001",
            "training.momentum=0.0",
            "attack.enabled=false",
            "runtime.device=cpu",
        ],
    )
    result = run_federated(config)

    assert result["rounds"] == 1


def test_centralized_run_with_sgd_optimizer(tmp_path):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path),
            "training.optimizer=sgd",
            "training.lr=0.001",
            "training.momentum=0.0",
            "tracking.enabled=false",
            "runtime.device=cpu",
        ],
    )
    result = run_centralized(config)

    assert "mse" in result
    assert "mae" in result
    assert "mape" in result


@pytest.mark.parametrize(
    "algorithm",
    [
        "fedavg",
        "adaptive_clipped_rdp_fedavg",
        "sparse_fedavg",
        "randomk_fedavg",
        "secure_quantized_fedavg",
        "sign_fedavg",
        "qsgd_fedavg",
        "ega_fedavg",
    ],
)
def test_validate_transport_modes_accepts_fixed_semantics_for_all_local_algorithms(tmp_path, algorithm):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            f"experiment.output_dir={tmp_path / algorithm}",
            f"federated.algorithm={algorithm}",
        ],
    )

    validate_transport_modes(config)


def test_validate_transport_modes_rejects_legacy_transport_keys():
    with pytest.raises(ValueError, match=r"transport\.upload_mode"):
        validate_transport_modes({"transport": {"upload_mode": "update"}})
    with pytest.raises(ValueError, match=r"transport\.download_mode"):
        validate_transport_modes({"transport": {"download_mode": "model"}})


class _IdentityEgaCodec(torch.nn.Module):
    """Minimal codec used to make EGA transport semantics deterministic in tests."""

    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = int(block_size)
        self.encoded_dim = int(block_size)
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        return blocks.to(torch.float32)

    def decode_blocks(self, encoded_blocks: torch.Tensor) -> torch.Tensor:
        return encoded_blocks.to(torch.float32)


def _run_manual_ega_rounds(config: dict, monkeypatch, load_calls=None):
    config.setdefault("ega", {})["num_clients"] = 2

    def _build_toy_model(_config):
        return _TransportToyModel()

    def _fake_train_one_epoch(model, loader, optimizer, device):
        del loader, optimizer, device
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
            for buffer in model.buffers():
                if buffer.dtype.is_floating_point:
                    buffer.add_(1.0)
        return 0.0

    def _fake_load_ega_codec(config, device, num_clients, allow_pretrain):
        del device, num_clients
        if load_calls is not None:
            load_calls.append(bool(allow_pretrain))
        block_size = int(config.get("ega", {}).get("block_size", 19))
        return _IdentityEgaCodec(block_size=block_size)

    def _fake_load_ega_codec_payload(config, payload, device, num_clients):
        del payload, device, num_clients
        block_size = int(config.get("ega", {}).get("block_size", 19))
        return _IdentityEgaCodec(block_size=block_size)

    monkeypatch.setattr(client_module, "build_model", _build_toy_model)
    monkeypatch.setattr(server_module, "build_model", _build_toy_model)
    monkeypatch.setattr(encoded_methods, "build_model", _build_toy_model)
    monkeypatch.setattr(client_module, "train_one_epoch", _fake_train_one_epoch)
    monkeypatch.setattr(encoded_methods, "load_ega_codec", _fake_load_ega_codec)
    monkeypatch.setattr(encoded_methods, "load_ega_codec_payload", _fake_load_ega_codec_payload)

    train_loaders = {"c1": _ToyLoader(), "c2": _ToyLoader()}
    val_loader = _ToyLoader(length=1)
    test_loader = _ToyLoader(length=1)
    device = torch.device("cpu")
    server = server_module.FederatedServer(config, val_loader, test_loader, device)
    total_train_samples = sum(len(loader.dataset) for loader in train_loaders.values())
    clients = [
        FederatedClient(
            client_id,
            loader,
            config,
            device,
            total_train_samples=total_train_samples,
            total_clients=len(train_loaders),
            allow_ega_pretrain=False,
        )
        for client_id, loader in train_loaders.items()
    ]

    rounds = []
    for round_index in range(3):
        round_base_state = _clone_state_dict(server.global_state)
        round_context = server.build_round_context()
        prepared_states = []
        results = []
        for client in clients:
            prepared = client.prepare_round_state(round_base_state, round_index=round_index, round_context=round_context)
            prepared_states.append((client.client_id, _clone_state_dict(prepared.download_state), _clone_state_dict(prepared.received_global_state)))
            result = client.train(
                round_base_state,
                round_index=round_index,
                round_context=round_context,
                prepared_state=prepared,
            )
            results.append(result)
        aggregation_weights = server.aggregate_dense(
            results,
            round_index=round_index,
            round_base_state=round_base_state,
            round_context=round_context,
        )
        rounds.append({
            "round_index": round_index,
            "round_context": round_context,
            "prepared_states": prepared_states,
            "results": results,
            "aggregation_weights": aggregation_weights,
            "global_state": _clone_state_dict(server.global_state),
        })
    return rounds


def test_ega_server_bootstraps_codec_once_via_round_context(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            f"experiment.output_dir={tmp_path / 'ega_bootstrap'}",
            "federated.algorithm=ega_fedavg",
            "federated.rounds=3",
            "training.epochs=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "data.shuffle_train=false",
            "training.optimizer=sgd",
            "training.lr=0.1",
            "ega.block_size=19",
            "ega.encoded_dim=19",
            "ega.hidden_dim=19",
            "ega.residual_blocks=0",
            "ega.quantization_level=64",
            "ega.encoded_dtype=float32",
            "ega.error_feedback=false",
            "ega.min_normalization=1e-6",
        ],
    )

    load_calls = []
    rounds = _run_manual_ega_rounds(config, monkeypatch, load_calls=load_calls)

    assert load_calls == [True]
    assert "ega_codec_payload" in rounds[0]["round_context"]
    assert "state_dict" in rounds[0]["round_context"]["ega_codec_payload"]
    assert all("ega_codec_payload" not in round_result["round_context"] for round_result in rounds[1:])


def test_ega_fedavg_transport_semantics_match_expected_received_models(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            f"experiment.output_dir={tmp_path / 'ega_fixed'}",
            "federated.algorithm=ega_fedavg",
            "federated.rounds=3",
            "training.epochs=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "data.shuffle_train=false",
            "training.optimizer=sgd",
            "training.lr=0.1",
            "ega.block_size=19",
            "ega.encoded_dim=19",
            "ega.hidden_dim=19",
            "ega.residual_blocks=0",
            "ega.quantization_level=64",
            "ega.encoded_dtype=float32",
            "ega.error_feedback=false",
            "ega.min_normalization=1e-6",
        ],
    )

    rounds = _run_manual_ega_rounds(config, monkeypatch)

    for round_result in rounds:
        expected_received = float(round_result["round_index"])
        for _, download_state, received_state in round_result["prepared_states"]:
            _assert_selected_values(
                download_state,
                {
                    "bn.weight": torch.tensor([expected_received, expected_received]),
                    "bn.running_mean": torch.tensor([expected_received, expected_received]),
                    "linear.weight": torch.full((2, 2), expected_received),
                },
            )
            _assert_selected_values(
                received_state,
                {
                    "bn.weight": torch.tensor([expected_received, expected_received]),
                    "bn.running_mean": torch.tensor([expected_received, expected_received]),
                    "linear.weight": torch.full((2, 2), expected_received),
                },
            )
        for result in round_result["results"]:
            assert result.aggregation_payload_kind == "ega_encoded_update"
            assert result.ega_payload is not None
            assert result.parameter_upload_bytes > 0
            assert result.parameter_upload_bytes == result.ega_payload.algorithm_num_bytes + client_module.state_num_bytes(result.aggregation_state)
            assert result.parameter_upload_parameters == result.ega_payload.algorithm_num_parameters + client_module.state_num_parameters(result.aggregation_state)
            assert result.parameter_upload_bytes > result.ega_payload.nbytes + client_module.state_num_bytes(result.aggregation_state)
        expected_global = float(round_result["round_index"] + 1)
        _assert_selected_values(
            round_result["global_state"],
            {
                "bn.weight": torch.tensor([expected_global, expected_global]),
                "bn.running_mean": torch.tensor([expected_global, expected_global]),
                "linear.weight": torch.full((2, 2), expected_global),
            },
        )
        assert round_result["aggregation_weights"] == [0.5, 0.5]


def test_single_node_transport_envelope_bytes_exceed_parameter_payload_for_dense_rounds(tmp_path, monkeypatch):
    config = _transport_test_config(tmp_path / "transport_dense")
    rounds = _run_manual_transport_rounds(config, monkeypatch)

    for round_result in rounds:
        for result in round_result["results"]:
            assert result.transport_download_bytes > result.parameter_download_bytes
            assert result.transport_upload_bytes > result.parameter_upload_bytes
            assert result.transport_download_overhead_bytes > 0
            assert result.transport_upload_overhead_bytes > 0


def test_auxiliary_payload_bytes_exclude_string_metadata():
    payload = {"label": "int8", "shape": [2, 3], "scale": 1.5, "flag": True}

    assert auxiliary_payload_num_bytes(payload) == 21
    assert auxiliary_payload_num_parameters(payload) == 4


def test_auxiliary_payload_counters_reject_unknown_objects():
    with pytest.raises(TypeError, match="Unsupported auxiliary payload type"):
        auxiliary_payload_num_bytes(object())
    with pytest.raises(TypeError, match="Unsupported auxiliary payload type"):
        auxiliary_payload_num_parameters(object())


def test_single_node_ega_transport_counts_round_context_bytes(tmp_path, monkeypatch):
    config = load_config(
        Path(__file__).parents[2] / "configs" / "test.yaml",
        [
            "experiment.output_dir=" + str(tmp_path / "ega_round_context"),
            "federated.algorithm=ega_fedavg",
            "federated.rounds=1",
            "training.epochs=1",
            "attack.enabled=false",
            "tracking.enabled=false",
            "runtime.device=cpu",
            "runtime.seed=2026",
            "data.shuffle_train=false",
            "training.optimizer=sgd",
            "training.lr=0.1",
            "ega.block_size=19",
            "ega.encoded_dim=19",
            "ega.hidden_dim=19",
            "ega.residual_blocks=0",
            "ega.quantization_level=64",
            "ega.encoded_dtype=float32",
            "ega.error_feedback=false",
            "ega.min_normalization=1e-6",
        ],
    )

    rounds = _run_manual_ega_rounds(config, monkeypatch)
    _, first_download_state, _ = rounds[0]["prepared_states"][0]
    _, second_download_state, _ = rounds[1]["prepared_states"][0]
    first_result = rounds[0]["results"][0]
    second_result = rounds[1]["results"][0]
    first_context = rounds[0]["round_context"]
    second_context = rounds[1]["round_context"]
    first_without_context = estimate_download_transport_bytes(first_download_state, round_index=0, compressed=False, round_context={})
    second_without_context = estimate_download_transport_bytes(second_download_state, round_index=1, compressed=False, round_context={})
    first_context_bytes = auxiliary_payload_num_bytes(first_context)
    first_context_parameters = auxiliary_payload_num_parameters(first_context)
    second_context_bytes = auxiliary_payload_num_bytes(second_context)
    second_context_parameters = auxiliary_payload_num_parameters(second_context)

    assert "ega_codec_payload" in first_context
    assert "ega_codec_payload" not in second_context
    assert first_result.parameter_download_bytes == first_result.download_bytes
    assert first_result.parameter_download_bytes == client_module.state_num_bytes(first_download_state) + first_context_bytes
    assert first_result.parameter_download_parameters == first_result.download_parameters
    assert first_result.parameter_download_parameters == client_module.state_num_parameters(first_download_state) + first_context_parameters
    assert second_result.parameter_download_bytes == client_module.state_num_bytes(second_download_state) + second_context_bytes
    assert second_result.parameter_download_parameters == client_module.state_num_parameters(second_download_state) + second_context_parameters
    assert first_result.parameter_download_bytes > second_result.parameter_download_bytes
    assert first_result.transport_download_bytes > first_result.parameter_download_bytes
    assert first_result.transport_download_bytes > first_without_context
    assert second_result.transport_download_bytes > second_result.parameter_download_bytes
    assert second_result.transport_download_bytes > second_without_context
    assert first_result.transport_download_overhead_bytes == first_result.transport_download_bytes - first_result.parameter_download_bytes
    assert second_result.transport_download_overhead_bytes == second_result.transport_download_bytes - second_result.parameter_download_bytes


class _ScalarTransportToyModel(torch.nn.Module):
    """Single-scalar-per-tensor model for exact update/update transport checks."""

    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1)
        self.linear = torch.nn.Linear(1, 1, bias=True)
        self.norm = torch.nn.LayerNorm(1)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()
            for buffer in self.buffers():
                if buffer.dtype.is_floating_point:
                    buffer.zero_()
                else:
                    buffer.zero_()

    def forward(self, x):
        return self.norm(self.linear(self.bn(x)))


def _single_node_update_update_config(tmp_path, algorithm: str, extra_overrides: list[str] | None = None) -> dict:
    overrides = [
        f"experiment.output_dir={tmp_path / algorithm}",
        f"federated.algorithm={algorithm}",
        "federated.rounds=3",
        "training.epochs=1",
        "attack.enabled=false",
        "tracking.enabled=false",
        "runtime.device=cpu",
        "runtime.seed=2026",
        "runtime.deterministic=true",
        "data.shuffle_train=false",
        "training.optimizer=sgd",
        "training.lr=0.1",
        "training.momentum=0.0",
        "training.weight_decay=0.0",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)
    return load_config(Path(__file__).parents[2] / "configs" / "test.yaml", overrides)


def _canonical_update_from_result(result, server):
    if result.sparse_update is not None:
        update = decompress_topk(result.sparse_update)
        if result.aggregation_state is not None:
            update.update(result.aggregation_state)
        return update
    if result.ega_payload is not None:
        update = encoded_methods.decode_mean_encoded_payload([result.ega_payload], server.ega_codec)
        if result.aggregation_state is not None:
            update.update(result.aggregation_state)
        return update
    if result.aggregation_state is None:
        raise AssertionError(f"Client {result.client_id} produced no aggregation payload")
    if result.aggregation_payload_kind in {"quantized_update", "sign_update"}:
        return dequantize_state_update(result.aggregation_state)
    if result.aggregation_payload_kind == "qsgd_update":
        levels = int(server.config.get("federated", {}).get("qsgd_levels", 127))
        return dequantize_qsgd_state_update(result.aggregation_state, levels=levels)
    return _clone_state_dict(result.aggregation_state)


def _run_single_node_update_update_rounds(config: dict, monkeypatch):
    config.setdefault("ega", {})["num_clients"] = 2

    def _build_toy_model(_config):
        return _ScalarTransportToyModel()

    def _fake_train_one_epoch(model, loader, optimizer, device):
        del loader, optimizer, device
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
            for buffer in model.buffers():
                if buffer.dtype.is_floating_point:
                    buffer.add_(1.0)
        return 0.0

    def _fake_load_ega_codec(config, device, num_clients, allow_pretrain):
        del device, num_clients, allow_pretrain
        block_size = int(config.get("ega", {}).get("block_size", 7))
        return _IdentityEgaCodec(block_size=block_size)

    def _fake_load_ega_codec_payload(config, payload, device, num_clients):
        del payload, device, num_clients
        block_size = int(config.get("ega", {}).get("block_size", 7))
        return _IdentityEgaCodec(block_size=block_size)

    monkeypatch.setattr(client_module, "build_model", _build_toy_model)
    monkeypatch.setattr(server_module, "build_model", _build_toy_model)
    monkeypatch.setattr(client_module, "train_one_epoch", _fake_train_one_epoch)
    if str(config.get("federated", {}).get("algorithm", "")).lower() == "ega_fedavg":
        monkeypatch.setattr(encoded_methods, "build_model", _build_toy_model)
        monkeypatch.setattr(encoded_methods, "load_ega_codec", _fake_load_ega_codec)
        monkeypatch.setattr(encoded_methods, "load_ega_codec_payload", _fake_load_ega_codec_payload)

    train_loaders = {"c1": _ToyLoader(), "c2": _ToyLoader()}
    val_loader = _ToyLoader(length=1)
    test_loader = _ToyLoader(length=1)
    device = torch.device("cpu")
    server = server_module.FederatedServer(config, val_loader, test_loader, device)
    total_train_samples = sum(len(loader.dataset) for loader in train_loaders.values())
    clients = [
        FederatedClient(
            client_id,
            loader,
            config,
            device,
            total_train_samples=total_train_samples,
            total_clients=len(train_loaders),
            allow_ega_pretrain=False,
        )
        for client_id, loader in train_loaders.items()
    ]

    rounds = []
    for round_index in range(3):
        round_base_state = _clone_state_dict(server.global_state)
        round_context = server.build_round_context()
        prepared_states = []
        results = []
        for client in clients:
            prepared = client.prepare_round_state(round_base_state, round_index=round_index, round_context=round_context)
            prepared_states.append(
                (
                    client.client_id,
                    _clone_state_dict(prepared.download_state),
                    _clone_state_dict(prepared.received_global_state),
                )
            )
            result = client.train(
                round_base_state,
                round_index=round_index,
                round_context=round_context,
                prepared_state=prepared,
            )
            results.append(result)
        aggregation_weights = server.aggregate_dense(
            results,
            round_index=round_index,
            round_base_state=round_base_state,
            round_context=round_context,
        )
        rounds.append(
            {
                "round_index": round_index,
                "prepared_states": prepared_states,
                "results": results,
                "aggregation_weights": aggregation_weights,
                "global_state": _clone_state_dict(server.global_state),
                "canonical_updates": [_canonical_update_from_result(result, server) for result in results],
            }
        )
    return rounds


@pytest.mark.parametrize(
    ("algorithm", "extra_overrides"),
    [
        ("fedavg", []),
                ("adaptive_clipped_rdp_fedavg", ["adaptive_clipped_rdp.clip_factor=1.0", "adaptive_clipped_rdp.min_clip_norm=100.0", "adaptive_clipped_rdp.max_clip_norm=100.0", "adaptive_clipped_rdp.reference_clip_norm=100.0", "adaptive_clipped_rdp.noise_multiplier=0.0"]),
                ("sparse_fedavg", ["federated.topk_fraction=1.0"]),
                ("randomk_fedavg", ["federated.topk_fraction=1.0", "federated.randomk_seed=2026"]),
                ("secure_quantized_fedavg", ["federated.quantization_dtype=float16", "privacy.clip_norm=0.0", "privacy.noise_multiplier=0.0"]),
        ("sign_fedavg", []),
        ("qsgd_fedavg", ["federated.qsgd_levels=127", "federated.quantization_seed=2026"]),
    ],
)
def test_single_node_sync_update_update_simulation_matches_expected_state_progression(tmp_path, monkeypatch, algorithm, extra_overrides):
    config = _single_node_update_update_config(tmp_path, algorithm, extra_overrides)

    rounds = _run_single_node_update_update_rounds(config, monkeypatch)

    expected_download_payload_values = [0.0, 1.0, 2.0]
    expected_received_values = [0.0, 1.0, 2.0]
    expected_global_values = [1.0, 2.0, 3.0]

    for round_result, expected_payload, expected_received, expected_global in zip(
        rounds,
        expected_download_payload_values,
        expected_received_values,
        expected_global_values,
    ):
        for _, download_state, received_state in round_result["prepared_states"]:
            if algorithm in {"fedavg", "adaptive_clipped_rdp_fedavg"}:
                _assert_state_float_value(download_state, expected_payload)
            _assert_state_float_value(received_state, expected_received)
        for update in round_result["canonical_updates"]:
            _assert_state_float_value(update, 1.0)
        _assert_state_float_value(round_result["global_state"], expected_global)
        assert round_result["aggregation_weights"] == [0.5, 0.5]


def _run_single_node_update_update_rounds_with_server(config: dict, monkeypatch):
    config.setdefault("ega", {})["num_clients"] = 2

    def _build_toy_model(_config):
        return _ScalarTransportToyModel()

    def _fake_train_one_epoch(model, loader, optimizer, device):
        del loader, optimizer, device
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
            for buffer in model.buffers():
                if buffer.dtype.is_floating_point:
                    buffer.add_(1.0)
        return 0.0

    def _fake_load_ega_codec(config, device, num_clients, allow_pretrain):
        del device, num_clients, allow_pretrain
        block_size = int(config.get("ega", {}).get("block_size", 7))
        return _IdentityEgaCodec(block_size=block_size)

    def _fake_load_ega_codec_payload(config, payload, device, num_clients):
        del payload, device, num_clients
        block_size = int(config.get("ega", {}).get("block_size", 7))
        return _IdentityEgaCodec(block_size=block_size)

    monkeypatch.setattr(client_module, "build_model", _build_toy_model)
    monkeypatch.setattr(server_module, "build_model", _build_toy_model)
    monkeypatch.setattr(client_module, "train_one_epoch", _fake_train_one_epoch)
    if str(config.get("federated", {}).get("algorithm", "")).lower() == "ega_fedavg":
        monkeypatch.setattr(encoded_methods, "build_model", _build_toy_model)
        monkeypatch.setattr(encoded_methods, "load_ega_codec", _fake_load_ega_codec)
        monkeypatch.setattr(encoded_methods, "load_ega_codec_payload", _fake_load_ega_codec_payload)

    train_loaders = {"c1": _ToyLoader(), "c2": _ToyLoader()}
    val_loader = _ToyLoader(length=1)
    test_loader = _ToyLoader(length=1)
    device = torch.device("cpu")
    server = server_module.FederatedServer(config, val_loader, test_loader, device)
    total_train_samples = sum(len(loader.dataset) for loader in train_loaders.values())
    clients = [
        FederatedClient(
            client_id,
            loader,
            config,
            device,
            total_train_samples=total_train_samples,
            total_clients=len(train_loaders),
            allow_ega_pretrain=False,
        )
        for client_id, loader in train_loaders.items()
    ]

    rounds = []
    for round_index in range(3):
        round_base_state = _clone_state_dict(server.global_state)
        round_context = server.build_round_context()
        prepared_states = []
        results = []
        for client in clients:
            prepared = client.prepare_round_state(round_base_state, round_index=round_index, round_context=round_context)
            prepared_states.append(
                (
                    client.client_id,
                    _clone_state_dict(prepared.download_state),
                    _clone_state_dict(prepared.received_global_state),
                )
            )
            result = client.train(
                round_base_state,
                round_index=round_index,
                round_context=round_context,
                prepared_state=prepared,
            )
            results.append(result)
        aggregation_weights = server.aggregate_dense(
            results,
            round_index=round_index,
            round_base_state=round_base_state,
            round_context=round_context,
        )
        rounds.append(
            {
                "round_index": round_index,
                "prepared_states": prepared_states,
                "results": results,
                "aggregation_weights": aggregation_weights,
                "global_state": _clone_state_dict(server.global_state),
                "canonical_updates": [_canonical_update_from_result(result, server) for result in results],
            }
        )
    return server, rounds
