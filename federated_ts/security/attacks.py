"""DLG and iDLG-style gradient reconstruction attacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import StateDict, load_serialized


@dataclass
class AttackResult:
    """Outcome of one gradient reconstruction attack.

    Example:
        ``AttackResult("DLG", 0.2, False)`` records a failed attack.
    """

    name: str
    reconstruction_mse: float
    success: bool


def _gradient_distance(model: nn.Module, x: torch.Tensor, y: torch.Tensor, target_grads: list[torch.Tensor]) -> torch.Tensor:
    """Compute squared distance between dummy and intercepted gradients."""

    criterion = nn.MSELoss()
    pred = model(x)
    loss = criterion(pred, y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True)
    return sum(torch.mean((grad - target.to(grad.device)) ** 2) for grad, target in zip(grads, target_grads))


def dlg_attack(
    config: dict[str, Any],
    state: StateDict,
    target_grads: list[torch.Tensor],
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
) -> AttackResult:
    """Reconstruct a batch by optimizing dummy inputs against observed gradients."""

    attack_cfg = config.get("attack", {})
    steps = int(attack_cfg.get("steps", 30))
    lr = float(attack_cfg.get("lr", 0.1))
    threshold = float(attack_cfg.get("success_mse_threshold", 1e-4))
    model = build_model(config).to(device)
    load_serialized(model, state, device)
    for param in model.parameters():
        param.requires_grad_(True)
    dummy_x = torch.randn_like(real_x, device=device, requires_grad=True)
    dummy_y = torch.randn_like(real_y, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_x, dummy_y], lr=lr)
    target = [grad.to(device) for grad in target_grads]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        dist = _gradient_distance(model, dummy_x, dummy_y, target)
        dist.backward()
        optimizer.step()
    rec_mse = torch.mean((dummy_x.detach().cpu() - real_x) ** 2).item()
    return AttackResult("DLG", rec_mse, rec_mse <= threshold)


def idlg_attack(
    config: dict[str, Any],
    state: StateDict,
    target_grads: list[torch.Tensor],
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
) -> AttackResult:
    """Run iDLG-style reconstruction.

    For regression forecasting there is no class label to infer from the final
    layer sign pattern, so this implementation fixes the target sequence to
    the intercepted batch target and optimizes only dummy inputs.
    """

    attack_cfg = config.get("attack", {})
    steps = int(attack_cfg.get("steps", 30))
    lr = float(attack_cfg.get("lr", 0.1))
    threshold = float(attack_cfg.get("success_mse_threshold", 1e-4))
    model = build_model(config).to(device)
    load_serialized(model, state, device)
    dummy_x = torch.randn_like(real_x, device=device, requires_grad=True)
    fixed_y = real_y.to(device)
    optimizer = torch.optim.Adam([dummy_x], lr=lr)
    target = [grad.to(device) for grad in target_grads]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        dist = _gradient_distance(model, dummy_x, fixed_y, target)
        dist.backward()
        optimizer.step()
    rec_mse = torch.mean((dummy_x.detach().cpu() - real_x) ** 2).item()
    return AttackResult("iDLG", rec_mse, rec_mse <= threshold)


def attack_success_rate(results: list[AttackResult]) -> float:
    """Compute the fraction of successful attacks.

    Example:
        ``attack_success_rate([AttackResult("DLG", 0.0, True)]) == 1.0``.
    """

    if not results:
        return 0.0
    return sum(result.success for result in results) / len(results)

