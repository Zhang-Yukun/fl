"""Attack execution helpers shared by training and offline replay."""

from __future__ import annotations

import time
from typing import Any

import torch
from loguru import logger

from fedlab.attack.tasks import AttackRoundResult, AttackRoundTask
from fedlab.security.attack_common import attach_attack_metadata
from fedlab.security.registry import run_attacks


def resolve_attack_device(config: dict[str, Any]) -> torch.device:
    """Resolve the device used for reconstruction attacks."""

    attack_cfg = config.get("attack", {})
    requested = str(attack_cfg.get("device", "same")).lower()
    if requested == "same":
        requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Attack device {} unavailable; falling back to CPU", requested)
        requested = "cpu"
    return torch.device(requested)


def _resolve_attack_max_samples(config: dict[str, Any], available_samples: int | None = None) -> int:
    """Resolve the explicit reconstruction count and reject the removed auto mode."""

    del available_samples
    configured = config.get("attack", {}).get("max_samples")
    if configured is None or str(configured).strip().lower() == "auto":
        raise ValueError("attack.max_samples must be an explicit positive integer; auto is no longer supported")
    count = int(configured)
    if count <= 0:
        raise ValueError("attack.max_samples must be a positive integer")
    return count


def _shape_tuple(value: Any, *, field: str) -> tuple[int, ...]:
    """Validate one configured per-sample tensor shape."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty list of positive integers")
    shape = tuple(int(dimension) for dimension in value)
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{field} must contain only positive integers")
    return shape


def _resolve_attack_shapes(config: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Resolve per-sample attack tensor shapes from explicit or task configuration."""

    attack_cfg = config.get("attack", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    task_type = str(config.get("task", {}).get("type", "forecasting")).lower()
    explicit_input = attack_cfg.get("input_shape")
    explicit_target = attack_cfg.get("target_shape")
    if explicit_input is not None:
        input_shape = _shape_tuple(explicit_input, field="attack.input_shape")
    elif task_type == "classification":
        input_shape = _shape_tuple(data_cfg.get("image_shape"), field="data.image_shape")
    elif task_type == "forecasting":
        input_shape = (
            int(data_cfg.get("seq_len", 21)),
            int(data_cfg.get("channels", model_cfg.get("channels", 1))),
        )
    else:
        raise ValueError(f"attack.input_shape is required for unsupported task.type={task_type!r}")

    if explicit_target is not None:
        target_shape = _shape_tuple(explicit_target, field="attack.target_shape")
    elif task_type == "classification":
        target_shape = None
    else:
        target_shape = (
            int(data_cfg.get("pred_len", 7)),
            int(data_cfg.get("channels", model_cfg.get("channels", 1))),
        )
    return input_shape, target_shape


def _build_attack_placeholders(config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create attack-only placeholders without loading any client training data."""

    count = _resolve_attack_max_samples(config)
    input_shape, target_shape = _resolve_attack_shapes(config)
    sample_x = torch.zeros((count, *input_shape), dtype=torch.float32)
    if target_shape is None:
        sample_y = torch.zeros((count,), dtype=torch.long)
    else:
        sample_y = torch.zeros((count, *target_shape), dtype=torch.float32)
    return sample_x, sample_y


def execute_attack_round_task(
    config: dict[str, Any],
    task: AttackRoundTask,
    attack_device: torch.device,
) -> AttackRoundResult:
    """Run reconstruction attacks from frozen updates without reference data."""

    start = time.perf_counter()
    attacks = []
    sample_x, sample_y = _build_attack_placeholders(config)
    for sample in task.samples:
        for result in run_attacks(
            config,
            sample.round_base_state,
            sample.target,
            sample_x,
            sample_y,
            attack_device,
            target_type=sample.target_type,
        ):
            attacks.append(
                attach_attack_metadata(
                    result,
                    client_id=sample.client_id,
                    round_index=sample.round_index,
                    sample_index=sample.sample_index,
                )
            )
    return AttackRoundResult(
        round_index=task.round_index,
        time_seconds=time.perf_counter() - start,
        clients_this_round=task.clients_this_round,
        evaluations_per_client=task.evaluations_per_client,
        attacks=attacks,
    )
