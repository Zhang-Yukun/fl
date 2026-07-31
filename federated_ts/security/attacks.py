"""DLG and iDLG-style gradient reconstruction attacks."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import StateDict, load_serialized


@dataclass
class AttackResult:
    """Outcome of one gradient reconstruction attack.

    Example:
        ``AttackResult("DLG", 0.2, 15.0, 0.1, 300, 1.5, False, 0.01, 0.3)``
        records a failed attack with draft-aligned metrics.
    """

    name: str
    mse: float
    psnr: float
    ssim: float
    iterations: int
    time_seconds: float
    success: bool
    success_threshold: float
    gradient_mse: float

    @property
    def reconstruction_mse(self) -> float:
        """Backward-compatible alias for the reconstructed-input MSE."""

        return self.mse

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable attack record.

        Example:
            ``AttackResult(...).to_record()["reconstruction_mse"]`` is the
            same value as ``mse`` for compatibility with older result files.
        """

        record = asdict(self)
        record["reconstruction_mse"] = self.mse
        for key, value in list(record.items()):
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None
        return record


def _gradient_distance(model: nn.Module, x: torch.Tensor, y: torch.Tensor, target_grads: list[torch.Tensor]) -> torch.Tensor:
    """Compute squared distance between dummy and intercepted gradients."""

    criterion = nn.MSELoss()
    pred = model(x)
    loss = criterion(pred, y)
    grads = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True)
    return sum(torch.mean((grad - target.to(grad.device)) ** 2) for grad, target in zip(grads, target_grads))


def _prepare_attack_model(config: dict[str, Any], state: StateDict, device: torch.device) -> nn.Module:
    """Build the attacked model with the configured deterministic mode.

    Example:
        ``model = _prepare_attack_model(config, state, device)`` loads global
        weights and switches to eval mode when ``attack.model_mode=eval``.
    """

    attack_cfg = config.get("attack", {})
    seed = attack_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
    model = build_model(config).to(device)
    load_serialized(model, state, device)
    if str(attack_cfg.get("model_mode", "train")) == "eval":
        model.eval()
    else:
        model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def _attack_threshold(config: dict[str, Any]) -> float:
    """Return the normalized-MSE success threshold used by the draft protocol."""

    return float(config.get("attack", {}).get("success_mse_threshold", 0.01))


def _attack_data_range(config: dict[str, Any]) -> float:
    """Return the assumed normalized data range for PSNR computation."""

    value = float(config.get("attack", {}).get("data_range", 1.0))
    return value if value > 0 else 1.0


def _compute_psnr(mse: float, data_range: float = 1.0) -> float:
    """Compute peak signal-to-noise ratio from reconstruction MSE.

    Example:
        ``round(_compute_psnr(0.01, 1.0), 2) == 20.0``.
    """

    if mse <= 0:
        return float("inf")
    return float(20.0 * math.log10(data_range / math.sqrt(mse)))


def _compute_ssim(reconstructed: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute the simplified global SSIM used by the draft protocol."""

    recon = reconstructed.detach().cpu().float().reshape(-1)
    real = target.detach().cpu().float().reshape(-1)
    mu1, mu2 = recon.mean(), real.mean()
    sigma1, sigma2 = recon.std(unbiased=False), real.std(unbiased=False)
    sigma12 = ((recon - mu1) * (real - mu2)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    denom = (mu1.square() + mu2.square() + c1) * (sigma1.square() + sigma2.square() + c2)
    if torch.isclose(denom, torch.tensor(0.0, dtype=denom.dtype)):
        return 0.0
    value = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / denom
    return float(value.item())


def _evaluate_reconstruction(
    name: str,
    reconstructed_x: torch.Tensor,
    real_x: torch.Tensor,
    iterations: int,
    elapsed: float,
    threshold: float,
    gradient_mse: float,
    data_range: float,
) -> AttackResult:
    """Build a draft-aligned attack result from a reconstructed input."""

    rec_mse = torch.mean((reconstructed_x.detach().cpu() - real_x.detach().cpu()) ** 2).item()
    psnr = _compute_psnr(rec_mse, data_range)
    ssim = _compute_ssim(reconstructed_x, real_x, data_range)
    return AttackResult(
        name=name,
        mse=float(rec_mse),
        psnr=psnr,
        ssim=ssim,
        iterations=iterations,
        time_seconds=elapsed,
        success=bool(rec_mse <= threshold),
        success_threshold=threshold,
        gradient_mse=float(gradient_mse),
    )


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
    threshold = _attack_threshold(config)
    data_range = _attack_data_range(config)
    start_time = time.perf_counter()
    model = _prepare_attack_model(config, state, device)
    dummy_x = torch.randn_like(real_x, device=device, requires_grad=True)
    dummy_y = torch.randn_like(real_y, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_x, dummy_y], lr=lr)
    target = [grad.to(device) for grad in target_grads]
    best_gradient_mse = float("inf")
    best_x = dummy_x.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        dist = _gradient_distance(model, dummy_x, dummy_y, target)
        dist_value = float(dist.detach().cpu().item())
        if dist_value < best_gradient_mse:
            best_gradient_mse = dist_value
            best_x = dummy_x.detach().clone()
        dist.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start_time
    return _evaluate_reconstruction("DLG", best_x, real_x, steps, elapsed, threshold, best_gradient_mse, data_range)


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
    threshold = _attack_threshold(config)
    data_range = _attack_data_range(config)
    start_time = time.perf_counter()
    model = _prepare_attack_model(config, state, device)
    dummy_x = torch.randn_like(real_x, device=device, requires_grad=True)
    fixed_y = real_y.to(device)
    optimizer = torch.optim.Adam([dummy_x], lr=lr)
    target = [grad.to(device) for grad in target_grads]
    best_gradient_mse = float("inf")
    best_x = dummy_x.detach().clone()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        dist = _gradient_distance(model, dummy_x, fixed_y, target)
        dist_value = float(dist.detach().cpu().item())
        if dist_value < best_gradient_mse:
            best_gradient_mse = dist_value
            best_x = dummy_x.detach().clone()
        dist.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start_time
    return _evaluate_reconstruction("iDLG", best_x, real_x, steps, elapsed, threshold, best_gradient_mse, data_range)


def attack_success_rate(results: list[AttackResult], name: str | None = None) -> float:
    """Compute the fraction of successful attacks.

    Example:
        ``attack_success_rate([AttackResult("DLG", 0.0, float("inf"), 1.0, 1, 0.1, True, 0.01, 0.0)]) == 1.0``.
    """

    filtered = [result for result in results if name is None or result.name == name]
    if not filtered:
        return 0.0
    return sum(result.success for result in filtered) / len(filtered)


def _mean_finite(values: list[float]) -> float | None:
    """Return the mean of finite values, or None when all values are non-finite."""

    finite = [value for value in values if math.isfinite(value)]
    if not values:
        return 0.0
    if not finite:
        return None
    return sum(finite) / len(finite)


def summarize_attack_results(results: list[AttackResult], success_rate_threshold: float = 0.03) -> dict[str, Any]:
    """Summarize DLG/iDLG metrics according to the draft protocol.

    Example:
        ``summarize_attack_results([])["overall_success_rate"] == 0.0``.
    """

    methods: dict[str, dict[str, float | int | bool | None]] = {}
    for name in sorted({result.name for result in results}):
        subset = [result for result in results if result.name == name]
        total = len(subset)
        success_count = sum(result.success for result in subset)
        methods[name] = {
            "success_count": success_count,
            "total_count": total,
            "success_rate": success_count / total if total else 0.0,
            "avg_mse": _mean_finite([result.mse for result in subset]),
            "avg_psnr": _mean_finite([result.psnr for result in subset]),
            "avg_ssim": _mean_finite([result.ssim for result in subset]),
            "avg_time_seconds": _mean_finite([result.time_seconds for result in subset]),
            "passes": (success_count / total if total else 0.0) <= success_rate_threshold,
        }
    overall_success_rate = attack_success_rate(results)
    return {
        "success_rate_threshold": success_rate_threshold,
        "overall_success_rate": overall_success_rate,
        "overall_passes": overall_success_rate <= success_rate_threshold,
        "methods": methods,
    }
