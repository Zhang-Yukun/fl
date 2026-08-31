"""Attack execution helpers shared by training and offline replay."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch
from loguru import logger

from fedlab.attack.data import ClientReferenceSet
from fedlab.attack.tasks import AttackRoundResult, AttackRoundTask
from fedlab.security.attack_common import apply_set_recovery_metrics, attach_attack_metadata
from fedlab.security.registry import run_attacks


ReferenceLoader = Callable[[str], ClientReferenceSet]


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


def _inverse_plot_tensor(values: torch.Tensor, mean: list[float] | None, std: list[float] | None) -> torch.Tensor:
    """Restore one standardized tensor for visualization only."""

    if mean is None or std is None:
        return values.detach().cpu().clone()
    tensor = values.detach().cpu().to(torch.float32)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).reshape(1, 1, -1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).reshape(1, 1, -1)
    while mean_tensor.ndim < tensor.ndim:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    return tensor * std_tensor + mean_tensor


def _resolve_attack_max_samples(config: dict[str, Any], available_samples: int | None) -> int | None:
    """Resolve how many client-local samples one replay attack should reconstruct."""

    configured = config.get("attack", {}).get("max_samples", 1)
    if configured is None:
        return None
    if str(configured).strip().lower() == "auto":
        count = None if available_samples is None else max(1, int(available_samples))
        cap = config.get("attack", {}).get("max_samples_cap")
        if count is not None and cap is not None:
            count = min(count, max(1, int(cap)))
        return count
    return max(1, int(configured))


def _build_attack_placeholders(reference_set: ClientReferenceSet, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create zero-valued attack placeholders matching the replay reference-set layout."""

    available = int(reference_set.inputs.shape[0])
    if available <= 0:
        raise ValueError("Attack replay reference set is empty")
    if reference_set.targets is None:
        raise ValueError("Attack replay requires reference targets to rebuild attack placeholders")
    count = _resolve_attack_max_samples(config, available)
    if count is None:
        count = available
    count = min(available, max(1, int(count)))
    return torch.zeros_like(reference_set.inputs[:count]), torch.zeros_like(reference_set.targets[:count])


def execute_attack_round_task(
    config: dict[str, Any],
    task: AttackRoundTask,
    attack_device: torch.device,
    *,
    reference_loader: ReferenceLoader | None = None,
) -> AttackRoundResult:
    """Run one detached round of DLG/iDLG evaluation from frozen snapshots."""

    if reference_loader is None:
        raise ValueError("Offline attack replay requires a reference_loader to rebuild client-local attack inputs")
    start = time.perf_counter()
    attacks = []
    reference_cache: dict[str, ClientReferenceSet] = {}

    def get_reference_set(client_id: str) -> ClientReferenceSet:
        if client_id not in reference_cache:
            reference_cache[client_id] = reference_loader(client_id)
        return reference_cache[client_id]

    for sample in task.samples:
        reference_set = get_reference_set(sample.client_id)
        sample_x, sample_y = _build_attack_placeholders(reference_set, config)
        for result in run_attacks(
            config,
            sample.round_base_state,
            sample.target,
            sample_x,
            sample_y,
            attack_device,
            target_type=sample.target_type,
            reference_inputs=reference_set.inputs,
            reference_targets=reference_set.targets,
        ):
            result = attach_attack_metadata(
                result,
                client_id=sample.client_id,
                round_index=sample.round_index,
                sample_index=sample.sample_index,
            )
            plot_reference_x = getattr(result, "reference_x", None)
            plot_reference_y = getattr(result, "reference_y", None)
            result.plot_reference_x = None if plot_reference_x is None else _inverse_plot_tensor(plot_reference_x, reference_set.scale_mean, reference_set.scale_std)
            result.plot_reconstructed_x = _inverse_plot_tensor(result.reconstructed_x, reference_set.scale_mean, reference_set.scale_std)
            result.plot_reference_y = None if plot_reference_y is None else _inverse_plot_tensor(plot_reference_y, reference_set.scale_mean, reference_set.scale_std)
            result.plot_reconstructed_y = None if result.reconstructed_y is None else _inverse_plot_tensor(result.reconstructed_y, reference_set.scale_mean, reference_set.scale_std)
            attacks.append(result)
    grouped: dict[tuple[str | None, int | None, str], list[Any]] = {}
    for result in attacks:
        grouped.setdefault((result.client_id, result.round_index, result.name), []).append(result)
    for key, subset in grouped.items():
        client_id, _round_index, _name = key
        if client_id is None:
            continue
        reference_set = get_reference_set(client_id)
        apply_set_recovery_metrics(
            subset,
            reference_inputs=reference_set.inputs,
            reference_targets=reference_set.targets,
            config=config,
        )
        for result in subset:
            if result.reference_x is not None:
                result.plot_reference_x = _inverse_plot_tensor(result.reference_x, reference_set.scale_mean, reference_set.scale_std)
            if result.reference_y is not None:
                result.plot_reference_y = _inverse_plot_tensor(result.reference_y, reference_set.scale_mean, reference_set.scale_std)
    return AttackRoundResult(
        round_index=task.round_index,
        time_seconds=time.perf_counter() - start,
        clients_this_round=task.clients_this_round,
        evaluations_per_client=task.evaluations_per_client,
        attacks=attacks,
    )
