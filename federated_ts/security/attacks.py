"""DLG and iDLG-style reconstruction attacks for gradients and transmitted updates."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import StateDict, load_serialized


@dataclass
class AttackResult:
    """Outcome of one reconstruction attack.

    Example:
        ``AttackResult("DLG", 0.2, 15.0, 0.1, 300, 1.5, False, 0.01, 0.3, "update_payload")``
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
    target_type: str = "gradient"
    exact_target_mse: float | None = None
    nearest_client_train_mse: float | None = None
    nearest_client_train_indices: list[int] | None = None
    metric_name: str = "reconstruction_mse"

    @property
    def reconstruction_mse(self) -> float:
        """Backward-compatible alias for the primary reconstructed-input MSE."""

        return self.mse

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable attack record."""

        record = asdict(self)
        record["reconstruction_mse"] = self.mse
        record["objective_mse"] = self.gradient_mse
        for key, value in list(record.items()):
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None
        return record


def _normalize_target_type(target_type: str | None, config: dict[str, Any]) -> str:
    """Resolve the configured attack target type."""

    value = target_type or config.get("attack", {}).get("target_type", "update_payload")
    normalized = str(value).lower()
    if normalized not in {"gradient", "update_payload"}:
        raise ValueError(f"Unsupported attack target type: {value}")
    return normalized


def _normalize_reference_metric(config: dict[str, Any], target_type: str) -> str:
    """Resolve the metric used to judge attack success."""

    value = str(config.get("attack", {}).get("reference_metric", "auto")).lower()
    if value == "auto":
        if target_type == "update_payload":
            return "nearest_client_train_mse"
        return "reconstruction_mse"
    if value not in {"reconstruction_mse", "nearest_client_train_mse"}:
        raise ValueError(f"Unsupported attack reference metric: {value}")
    return value


def _gradient_distance(model: nn.Module, x: torch.Tensor, y: torch.Tensor, target_grads: list[torch.Tensor]) -> torch.Tensor:
    """Compute squared distance between dummy and intercepted gradients."""

    criterion = nn.MSELoss()
    pred = model(x)
    loss = criterion(pred, y)
    trainable_parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    grads = torch.autograd.grad(loss, trainable_parameters, create_graph=True)
    return sum(torch.mean((grad - target.to(grad.device)) ** 2) for grad, target in zip(grads, target_grads))


def _predicted_trainable_update(model: nn.Module, x: torch.Tensor, y: torch.Tensor, config: dict[str, Any]) -> OrderedDict[str, torch.Tensor]:
    """Approximate the transmitted one-step local update for one dummy batch."""

    criterion = nn.MSELoss()
    pred = model(x)
    loss = criterion(pred, y)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    grads = torch.autograd.grad(loss, tuple(parameter for _, parameter in named_parameters), create_graph=True, allow_unused=True)
    attack_cfg = config.get("attack", {})
    optimizer_name = str(attack_cfg.get("local_optimizer", "adam")).lower()
    lr = float(attack_cfg.get("local_lr", config.get("training", {}).get("lr", 1e-3)))
    eps = float(attack_cfg.get("local_optimizer_eps", 1e-8))
    updates: OrderedDict[str, torch.Tensor] = OrderedDict()
    for (name, parameter), grad in zip(named_parameters, grads):
        if grad is None:
            grad = torch.zeros_like(parameter)
        if optimizer_name == "adam":
            update = -lr * grad / (torch.abs(grad) + eps)
        elif optimizer_name == "sgd":
            update = -lr * grad
        else:
            raise ValueError(f"Unsupported local optimizer for update attack: {optimizer_name}")
        updates[name] = update
    return updates


def _update_distance(model: nn.Module, x: torch.Tensor, y: torch.Tensor, target_update: StateDict, config: dict[str, Any]) -> torch.Tensor:
    """Compute squared distance between dummy one-step updates and an intercepted payload."""

    predicted = _predicted_trainable_update(model, x, y, config)
    distance = None
    for name, target in target_update.items():
        if name not in predicted:
            continue
        diff = predicted[name] - target.to(predicted[name].device, dtype=predicted[name].dtype)
        term = torch.mean(diff ** 2)
        distance = term if distance is None else distance + term
    if distance is None:
        raise ValueError("Attack target update does not overlap with any trainable model parameter")
    return distance


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
    """Return the normalized-MSE success threshold used by the attack rules."""

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


def _nearest_reference_mse(reconstructed: torch.Tensor, reference_inputs: torch.Tensor) -> tuple[float | None, list[int] | None]:
    """Return the average nearest-neighbor MSE against a reference window set."""

    references = reference_inputs.detach().cpu().float()
    if references.numel() == 0:
        return None, None
    recon = reconstructed.detach().cpu().float()
    recon_flat = recon.reshape(recon.shape[0], -1)
    refs_flat = references.reshape(references.shape[0], -1)
    mse_matrix = torch.mean((recon_flat[:, None, :] - refs_flat[None, :, :]) ** 2, dim=-1)
    nearest_values, nearest_indices = torch.min(mse_matrix, dim=1)
    return float(nearest_values.mean().item()), [int(index) for index in nearest_indices.tolist()]


def _evaluate_reconstruction(
    name: str,
    reconstructed_x: torch.Tensor,
    real_x: torch.Tensor,
    iterations: int,
    elapsed: float,
    threshold: float,
    objective_mse: float,
    data_range: float,
    ssim_threshold: float | None,
    target_type: str,
    reference_inputs: torch.Tensor | None = None,
    reference_metric: str = "reconstruction_mse",
) -> AttackResult:
    """Build an attack result from a reconstructed input."""

    exact_target_mse = torch.mean((reconstructed_x.detach().cpu() - real_x.detach().cpu()) ** 2).item()
    psnr = _compute_psnr(exact_target_mse, data_range)
    ssim = _compute_ssim(reconstructed_x, real_x, data_range)
    nearest_client_train_mse = None
    nearest_client_train_indices = None
    if reference_inputs is not None:
        nearest_client_train_mse, nearest_client_train_indices = _nearest_reference_mse(reconstructed_x, reference_inputs)
    metric_name = reference_metric
    primary_mse = float(exact_target_mse)
    if metric_name == "nearest_client_train_mse":
        if nearest_client_train_mse is None:
            metric_name = "reconstruction_mse"
        else:
            primary_mse = float(nearest_client_train_mse)
    success = primary_mse <= threshold
    if metric_name == "reconstruction_mse" and ssim_threshold is not None:
        success = success or ssim >= ssim_threshold
    return AttackResult(
        name=name,
        mse=primary_mse,
        psnr=psnr,
        ssim=ssim,
        iterations=iterations,
        time_seconds=elapsed,
        success=bool(success),
        success_threshold=threshold,
        gradient_mse=float(objective_mse),
        target_type=target_type,
        exact_target_mse=float(exact_target_mse),
        nearest_client_train_mse=None if nearest_client_train_mse is None else float(nearest_client_train_mse),
        nearest_client_train_indices=nearest_client_train_indices,
        metric_name=metric_name,
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
    target: list[torch.Tensor] | StateDict,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
    name: str,
    optimize_y: bool,
    target_type: str | None = None,
    reference_inputs: torch.Tensor | None = None,
) -> AttackResult:
    """Run one configurable reconstruction attack loop."""

    resolved_target_type = _normalize_target_type(target_type, config)
    resolved_reference_metric = _normalize_reference_metric(config, resolved_target_type)
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
    if resolved_target_type == "gradient":
        prepared_target = [grad.to(device) for grad in target]
    else:
        prepared_target = OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in target.items())
    overall_start = time.perf_counter()
    best_objective_mse = float("inf")
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
        restart_best_objective = float("inf")
        restart_best_x = dummy_x.detach().clone()

        for _ in range(steps):
            if optimizer_name == "lbfgs":
                holder: dict[str, float] = {}

                def closure() -> torch.Tensor:
                    """Evaluate the current dummy variables for one LBFGS step."""

                    optimizer.zero_grad(set_to_none=True)
                    if resolved_target_type == "gradient":
                        dist = _gradient_distance(model, dummy_x, dummy_y, prepared_target)
                    else:
                        dist = _update_distance(model, dummy_x, dummy_y, prepared_target, config)
                    if tv_weight > 0:
                        dist = dist + tv_weight * _time_series_total_variation(dummy_x)
                    holder["loss"] = float(dist.detach().cpu().item())
                    dist.backward()
                    return dist

                optimizer.step(closure)
                dist_value = holder.get("loss", float("inf"))
            else:
                optimizer.zero_grad(set_to_none=True)
                if resolved_target_type == "gradient":
                    dist = _gradient_distance(model, dummy_x, dummy_y, prepared_target)
                else:
                    dist = _update_distance(model, dummy_x, dummy_y, prepared_target, config)
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
            if dist_value < restart_best_objective:
                restart_best_objective = dist_value
                restart_best_x = dummy_x.detach().clone()
        if restart_best_objective < best_objective_mse:
            best_objective_mse = restart_best_objective
            best_x = restart_best_x
    elapsed = time.perf_counter() - overall_start
    return _evaluate_reconstruction(
        name,
        best_x,
        real_x,
        steps,
        elapsed,
        threshold,
        best_objective_mse,
        data_range,
        ssim_threshold,
        resolved_target_type,
        reference_inputs=reference_inputs,
        reference_metric=resolved_reference_metric,
    )


def dlg_attack(
    config: dict[str, Any],
    state: StateDict,
    target: list[torch.Tensor] | StateDict,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
    target_type: str | None = None,
    reference_inputs: torch.Tensor | None = None,
) -> AttackResult:
    """Reconstruct a batch by optimizing dummy inputs against the chosen target."""

    return _attack_loop(
        config,
        state,
        target,
        real_x,
        real_y,
        device,
        name="DLG",
        optimize_y=True,
        target_type=target_type,
        reference_inputs=reference_inputs,
    )


def idlg_attack(
    config: dict[str, Any],
    state: StateDict,
    target: list[torch.Tensor] | StateDict,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
    target_type: str | None = None,
    reference_inputs: torch.Tensor | None = None,
) -> AttackResult:
    """Run iDLG-style reconstruction for forecasting targets."""

    return _attack_loop(
        config,
        state,
        target,
        real_x,
        real_y,
        device,
        name="iDLG",
        optimize_y=False,
        target_type=target_type,
        reference_inputs=reference_inputs,
    )


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
    """Summarize DLG/iDLG metrics with the configured primary view."""

    primary_metric = results[0].metric_name if results else "reconstruction_mse"
    methods: dict[str, dict[str, float | int | bool | None | str]] = {}
    for name in sorted({result.name for result in results}):
        subset = [result for result in results if result.name == name]
        total = len(subset)
        success_count = sum(result.success for result in subset)
        methods[name] = {
            "primary_metric": subset[0].metric_name if subset else primary_metric,
            "target_type": subset[0].target_type if subset else None,
            "success_count": success_count,
            "total_count": total,
            "success_rate": success_count / total if total else 0.0,
            "success_rate_percent": round((success_count / total if total else 0.0) * 100.0, 2),
            "avg_mse": _mean_finite([result.mse for result in subset]),
            "best_mse": _mean_finite(sorted([result.mse for result in subset])[:1]),
            "avg_exact_target_mse": _mean_finite([result.exact_target_mse for result in subset if result.exact_target_mse is not None]),
            "avg_nearest_client_train_mse": _mean_finite([result.nearest_client_train_mse for result in subset if result.nearest_client_train_mse is not None]),
            "avg_psnr": _mean_finite([result.psnr for result in subset]),
            "avg_ssim": _mean_finite([result.ssim for result in subset]),
            "best_ssim": _mean_finite(sorted([result.ssim for result in subset], reverse=True)[:1]),
            "avg_gradient_mse": _mean_finite([result.gradient_mse for result in subset]),
            "avg_objective_mse": _mean_finite([result.gradient_mse for result in subset]),
            "avg_time_seconds": _mean_finite([result.time_seconds for result in subset]),
            "passes": (success_count / total if total else 0.0) <= success_rate_threshold,
        }
    overall_success_rate = attack_success_rate(results)
    overall_avg_mse = _mean_finite([result.mse for result in results])
    overall_best_mse = _mean_finite(sorted([result.mse for result in results])[:1])
    overall_avg_psnr = _mean_finite([result.psnr for result in results])
    overall_avg_ssim = _mean_finite([result.ssim for result in results])
    overall_avg_gradient_mse = _mean_finite([result.gradient_mse for result in results])
    overall_avg_exact_target_mse = _mean_finite([result.exact_target_mse for result in results if result.exact_target_mse is not None])
    overall_avg_nearest_client_train_mse = _mean_finite([result.nearest_client_train_mse for result in results if result.nearest_client_train_mse is not None])
    return {
        "primary_metric": primary_metric,
        "primary_metric_direction": "higher_is_more_private",
        "target_type": results[0].target_type if results else None,
        "success_rate_threshold": success_rate_threshold,
        "overall_avg_mse": overall_avg_mse,
        "overall_best_mse": overall_best_mse,
        "overall_avg_exact_target_mse": overall_avg_exact_target_mse,
        "overall_avg_nearest_client_train_mse": overall_avg_nearest_client_train_mse,
        "overall_avg_psnr": overall_avg_psnr,
        "overall_avg_ssim": overall_avg_ssim,
        "overall_avg_gradient_mse": overall_avg_gradient_mse,
        "overall_avg_objective_mse": overall_avg_gradient_mse,
        "overall_success_rate": overall_success_rate,
        "overall_success_rate_percent": round(overall_success_rate * 100.0, 2),
        "overall_passes": overall_success_rate <= success_rate_threshold,
        "methods": methods,
    }
