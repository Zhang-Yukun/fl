"""Attack replay task structures and builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from fedlab.utils.serialization import StateDict


@dataclass
class AttackSampleTask:
    """Immutable attack inputs captured from one round before async execution."""

    client_id: str
    round_index: int
    sample_index: int
    target_type: str
    round_base_state: StateDict
    target: list[torch.Tensor] | StateDict
    sample_x_shape: tuple[int, ...]
    sample_y_shape: tuple[int, ...]
    sample_x_dtype: str
    sample_y_dtype: str
    scale_mean: list[float] | None = None
    scale_std: list[float] | None = None


@dataclass
class AttackRoundTask:
    """One round of attack work detached from the training hot path."""

    round_index: int
    clients_this_round: int
    evaluations_per_client: int
    samples: list[AttackSampleTask]


@dataclass
class AttackRoundResult:
    """Completed attack artifacts for one training round."""

    round_index: int
    time_seconds: float
    clients_this_round: int
    evaluations_per_client: int
    attacks: list[Any]


def _clone_state(state: StateDict) -> StateDict:
    return type(state)((name, tensor.detach().cpu().clone()) for name, tensor in state.items())


def _clone_attack_target(target: list[torch.Tensor] | StateDict) -> list[torch.Tensor] | StateDict:
    if isinstance(target, list):
        return [tensor.detach().cpu().clone() for tensor in target]
    return _clone_state(target)


def _should_capture_update_payload(config: dict[str, Any], round_index: int, max_rounds: int) -> bool:
    capture_cfg = config.get("replay_capture", {})
    if not capture_cfg.get("enabled", True):
        return False
    frequency = int(capture_cfg.get("frequency_rounds", 30))
    return round_index == 0 or round_index == max_rounds - 1 or (frequency > 0 and round_index % frequency == 0)


def _should_run_attack(config: dict[str, Any], round_index: int, max_rounds: int) -> bool:
    attack_cfg = config.get("attack", {})
    if not attack_cfg.get("enabled", True):
        return False
    return _should_capture_update_payload(config, round_index, max_rounds)


def _select_attack_client_ids(client_ids: list[str], config: dict[str, Any], round_index: int) -> list[str]:
    if not client_ids:
        return []
    attack_cfg = config.get("attack", {})
    selection = str(attack_cfg.get("client_selection", "all")).lower()
    count = max(1, int(attack_cfg.get("clients_per_round", 1)))
    if selection == "all":
        return list(client_ids)
    if selection == "first":
        return list(client_ids[:count])
    start = round_index % len(client_ids)
    return [client_ids[(start + offset) % len(client_ids)] for offset in range(min(count, len(client_ids)))]


def build_update_attack_round_task(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    round_index: int,
    max_rounds: int,
) -> AttackRoundTask | None:
    """Build one update-payload attack task from previously captured update records."""

    if not _should_run_attack(config, round_index, max_rounds):
        return None
    records_this_round = [record for record in records if int(record["round_index"]) == int(round_index)]
    if not records_this_round:
        return None
    client_order = [str(client_id) for client_id in config.get("data", {}).get("clients", [])]
    order_index = {client_id: index for index, client_id in enumerate(client_order)}
    records_this_round = sorted(
        records_this_round,
        key=lambda record: (order_index.get(str(record["client_id"]), len(order_index)), str(record["client_id"])),
    )
    client_ids = [str(record["client_id"]) for record in records_this_round]
    selected_client_ids = set(_select_attack_client_ids(client_ids, config, round_index))
    selected_records = [record for record in records_this_round if str(record["client_id"]) in selected_client_ids]
    if not selected_records:
        return None
    samples: list[AttackSampleTask] = []
    evaluations_per_client = 0
    for record in selected_records:
        record_samples = list(record.get("samples", []))
        evaluations_per_client = max(evaluations_per_client, len(record_samples))
        for sample in record_samples:
            samples.append(
                AttackSampleTask(
                    client_id=str(record["client_id"]),
                    round_index=int(record["round_index"]),
                    sample_index=int(sample.get("sample_index", 0)),
                    target_type="update_payload",
                    round_base_state=_clone_state(record["round_base_state"]),
                    target=_clone_attack_target(record["target_update"]),
                    sample_x_shape=tuple(int(value) for value in sample["sample_x_shape"]),
                    sample_y_shape=tuple(int(value) for value in sample["sample_y_shape"]),
                    sample_x_dtype=str(sample["sample_x_dtype"]),
                    sample_y_dtype=str(sample["sample_y_dtype"]),
                    scale_mean=None if record.get("scale_mean") is None else [float(value) for value in record["scale_mean"]],
                    scale_std=None if record.get("scale_std") is None else [float(value) for value in record["scale_std"]],
                )
            )
    return AttackRoundTask(
        round_index=int(round_index),
        clients_this_round=len(selected_records),
        evaluations_per_client=evaluations_per_client,
        samples=samples,
    )
