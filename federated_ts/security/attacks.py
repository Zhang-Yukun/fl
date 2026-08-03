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
        records one reconstruction attempt with draft-aligned metrics.
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
        """Return a JSON-serializable attack record."""

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
    trainable_parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    grads = torch.autograd.grad(loss, trainable_parameters, create_graph=True)
    return sum(torch.mean((grad - target.to(grad.device)) ** 2) for grad, target in zip(grads, target_grads))


def _time_series_total_variation(x: torch.Tensor) -> torch.Tensor:
    """Return a small total-variation regularizer for 1D windows."""

    if x.ndim < 3 or x.shape[1] <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))


def _prepare_attack_model(config: dict[str, Any], state: StateDict, device: torch.device) -> nn.Module:
    """Build the attacked model with the configured deterministic mode."""

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
    return model


def _attack_threshold(config: dict[str, Any]) -> float:
    """Return the normalized-MSE success threshold used by the draft protocol."""

    return float(config.get("attack", {}).get("success_mse_threshold", 0.01))


def _attack_ssim_threshold(config: dict[str, Any]) -> float | None:
    """Return an optional structural-success threshold."""

    value = config.get("attack", {}).get("success_ssim_threshold")
    if value is None:
        return None
    return float(value)


def _attack_data_range(config: dict[str, Any]) -> float:
    """Return the assumed normalized data range for PSNR computation."""

    value = float(config.get("attack", {}).get("data_range", 1.0))
    return value if value > 0 else 1.0


def _compute_psnr(mse: float, data_range: float = 1.0) -> float:
    """Compute peak signal-to-noise ratio from reconstruction MSE."""

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
    ssim_threshold: float | None,
) -> AttackResult:
    """Build an attack result from a reconstructed input."""

    rec_mse = torch.mean((reconstructed_x.detach().cpu() - real_x.detach().cpu()) ** 2).item()
    psnr = _compute_psnr(rec_mse, data_range)
    ssim = _compute_ssim(reconstructed_x, real_x, data_range)
    success = rec_mse <= threshold
    if ssim_threshold is not None:
        success = success or ssim >= ssim_threshold
    return AttackResult(
        name=name,
        mse=float(rec_mse),
        psnr=psnr,
        ssim=ssim,
        iterations=iterations,
        time_seconds=elapsed,
        success=bool(success),
        success_threshold=threshold,
        gradient_mse=float(gradient_mse),
    )


def _create_optimizer(name: str, variables: list[torch.Tensor], lr: float, history_size: int) -> torch.optim.Optimizer:
    """Create the configured optimizer for reconstruction."""

    if name == "lbfgs":
        return torch.optim.LBFGS(variables, lr=lr, max_iter=1, history_size=history_size, line_search_fn="strong_wolfe")
    if name == "adam":
        return torch.optim.Adam(variables, lr=lr)
    raise ValueError(f"Unknown attack optimizer: {name}")


def _attack_loop(
    config: dict[str, Any],
    state: StateDict,
    target_grads: list[torch.Tensor],
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
    name: str,
    optimize_y: bool,
) -> AttackResult:
    """Run one configurable reconstruction attack loop."""

    attack_cfg = config.get("attack", {})
    steps = int(attack_cfg.get("steps", 300))
    lr = float(attack_cfg.get("lr", 0.1))
    threshold = _attack_threshold(config)
    ssim_threshold = _attack_ssim_threshold(config)
    data_range = _attack_data_range(config)
    optimizer_name = str(attack_cfg.get("optimizer", "adam")).lower()
    restarts = max(1, int(attack_cfg.get("restarts", 1)))
    history_size = int(attack_cfg.get("lbfgs_history_size", 20))
    input_clip = attack_cfg.get("input_clip")
    target_clip = attack_cfg.get("target_clip")
    tv_weight = float(attack_cfg.get("tv_weight", 0.0))
    seed = attack_cfg.get("seed")
    target = [grad.to(device) for grad in target_grads]
    overall_start = time.perf_counter()
    best_gradient_mse = float("inf")
    best_x = real_x.to(device)

    for restart in range(restarts):
        if seed is not None:
            local_seed = int(seed) + restart
            torch.manual_seed(local_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(local_seed)
        model = _prepare_attack_model(config, state, device)
        dummy_x = torch.randn_like(real_x, device=device, requires_grad=True)
        if optimize_y:
            dummy_y = torch.randn_like(real_y, device=device, requires_grad=True)
            variables = [dummy_x, dummy_y]
        else:
            dummy_y = real_y.to(device)
            variables = [dummy_x]
        optimizer = _create_optimizer(optimizer_name, variables, lr, history_size)
        restart_best_gradient = float("inf")
        restart_best_x = dummy_x.detach().clone()

        for _ in range(steps):
            if optimizer_name == "lbfgs":
                holder: dict[str, float] = {}

                def closure() -> torch.Tensor:
                    """Evaluate the current dummy variables for one LBFGS step."""

                    optimizer.zero_grad(set_to_none=True)
                    dist = _gradient_distance(model, dummy_x, dummy_y, target)
                    if tv_weight > 0:
                        dist = dist + tv_weight * _time_series_total_variation(dummy_x)
                    holder["loss"] = float(dist.detach().cpu().item())
                    dist.backward()
                    return dist

                optimizer.step(closure)
                dist_value = holder.get("loss", float("inf"))
            else:
                optimizer.zero_grad(set_to_none=True)
                dist = _gradient_distance(model, dummy_x, dummy_y, target)
                if tv_weight > 0:
                    dist = dist + tv_weight * _time_series_total_variation(dummy_x)
                dist_value = float(dist.detach().cpu().item())
                dist.backward()
                optimizer.step()
            with torch.no_grad():
                if input_clip is not None:
                    dummy_x.clamp_(min=-float(input_clip), max=float(input_clip))
                if optimize_y and target_clip is not None:
                    dummy_y.clamp_(min=-float(target_clip), max=float(target_clip))
            if dist_value < restart_best_gradient:
                restart_best_gradient = dist_value
                restart_best_x = dummy_x.detach().clone()
        if restart_best_gradient < best_gradient_mse:
            best_gradient_mse = restart_best_gradient
            best_x = restart_best_x
    elapsed = time.perf_counter() - overall_start
    return _evaluate_reconstruction(name, best_x, real_x, steps, elapsed, threshold, best_gradient_mse, data_range, ssim_threshold)


def dlg_attack(
    config: dict[str, Any],
    state: StateDict,
    target_grads: list[torch.Tensor],
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
) -> AttackResult:
    """Reconstruct a batch by optimizing dummy inputs against observed gradients."""

    return _attack_loop(config, state, target_grads, real_x, real_y, device, name="DLG", optimize_y=True)


def idlg_attack(
    config: dict[str, Any],
    state: StateDict,
    target_grads: list[torch.Tensor],
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
) -> AttackResult:
    """Run iDLG-style reconstruction for forecasting regression targets.

    For regression forecasting there is no class label to infer from the final
    layer sign pattern, so this implementation fixes the target sequence to the
    intercepted batch target and optimizes only dummy inputs.
    """

    return _attack_loop(config, state, target_grads, real_x, real_y, device, name="iDLG", optimize_y=False)


def attack_success_rate(results: list[AttackResult], name: str | None = None) -> float:
    """Compute the fraction of successful attacks."""

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
    """Summarize DLG/iDLG metrics according to the draft protocol."""

    methods: dict[str, dict[str, float | int | bool | None]] = {}
    for name in sorted({result.name for result in results}):
        subset = [result for result in results if result.name == name]
        total = len(subset)
        success_count = sum(result.success for result in subset)
        methods[name] = {
            "success_count": success_count,
            "total_count": total,
            "success_rate": success_count / total if total else 0.0,
            "success_rate_percent": round((success_count / total if total else 0.0) * 100.0, 2),
            "avg_mse": _mean_finite([result.mse for result in subset]),
            "best_mse": _mean_finite(sorted([result.mse for result in subset])[:1]),
            "avg_psnr": _mean_finite([result.psnr for result in subset]),
            "avg_ssim": _mean_finite([result.ssim for result in subset]),
            "best_ssim": _mean_finite(sorted([result.ssim for result in subset], reverse=True)[:1]),
            "avg_gradient_mse": _mean_finite([result.gradient_mse for result in subset]),
            "avg_time_seconds": _mean_finite([result.time_seconds for result in subset]),
            "passes": (success_count / total if total else 0.0) <= success_rate_threshold,
        }
    overall_success_rate = attack_success_rate(results)
    return {
        "success_rate_threshold": success_rate_threshold,
        "overall_success_rate": overall_success_rate,
        "overall_success_rate_percent": round(overall_success_rate * 100.0, 2),
        "overall_passes": overall_success_rate <= success_rate_threshold,
        "methods": methods,
    }
