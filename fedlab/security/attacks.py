"""DLG and iDLG-style reconstruction attacks for gradients and transmitted updates."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fedlab.modeling import build_model
from fedlab.tasks import create_loss
from fedlab.utils.serialization import StateDict, load_serialized


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
    client_id: str | None = None
    round_index: int | None = None
    sample_index: int | None = None
    artifact_path: str | None = None
    real_x: torch.Tensor | None = None
    real_y: torch.Tensor | None = None
    reference_x: torch.Tensor | None = None
    reference_y: torch.Tensor | None = None
    reference_label: str | None = None
    reconstructed_x: torch.Tensor | None = None
    reconstructed_y: torch.Tensor | None = None

    @property
    def reconstruction_mse(self) -> float:
        """Backward-compatible alias for the primary reconstructed-input MSE."""

        return self.mse

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable attack record."""

        record = asdict(self)
        for key in ("real_x", "real_y", "reference_x", "reference_y", "reconstructed_x", "reconstructed_y"):
            record.pop(key, None)
        record["primary_metric_name"] = record.pop("metric_name")
        record["primary_metric_value"] = record.pop("mse")
        record["objective_mse"] = record.pop("gradient_mse")
        for key, value in list(record.items()):
            if value is None:
                record.pop(key, None)
                continue
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


def _normalize_report_metrics(config: dict[str, Any], target_type: str) -> set[str]:
    """Resolve which auxiliary attack metrics should be exposed in outputs."""

    value = config.get("attack", {}).get("report_metrics", "auto")
    if value == "auto" or value is None:
        return {"nearest_client_train_mse"} if target_type == "update_payload" else {"exact_target_mse"}
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in value if str(item).strip()]
    allowed = {"exact_target_mse", "nearest_client_train_mse"}
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unsupported attack report_metrics entries: {unknown}")
    return set(items)


def _gradient_distance(model: nn.Module, x: torch.Tensor, y: torch.Tensor, target_grads: list[torch.Tensor], config: dict[str, Any]) -> torch.Tensor:
    """Compute squared distance between dummy and intercepted gradients."""

    criterion = create_loss(config)
    pred = model(x)
    loss = criterion(pred, y)
    trainable_parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    grads = torch.autograd.grad(loss, trainable_parameters, create_graph=True)
    return sum(torch.mean((grad - target.to(grad.device)) ** 2) for grad, target in zip(grads, target_grads))


def _predicted_trainable_update(model: nn.Module, x: torch.Tensor, y: torch.Tensor, config: dict[str, Any]) -> OrderedDict[str, torch.Tensor]:
    """Approximate the transmitted one-step local update for one dummy batch."""

    criterion = create_loss(config)
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

    model = build_model(config).to(device)
    load_serialized(model, state, device)
    attack_cfg = config.get("attack", {})
    if str(attack_cfg.get("model_mode", "train")) == "eval":
        model.eval()
    else:
        model.train()
    return model


def _attack_rng_context(device: torch.device):
    """Fork torch RNG state so async attack threads do not leak seeds into training."""

    if device.type != "cuda":
        return torch.random.fork_rng(devices=[])
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.random.fork_rng(devices=[device_index])


def _attack_threshold(config: dict[str, Any]) -> float:
    """Return the normalized-MSE success threshold used by the attack rules."""

    return float(config.get("attack", {}).get("success_mse_threshold", 0.5))


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


def _select_reference_targets(
    reference_inputs: torch.Tensor | None,
    reference_targets: torch.Tensor | None,
    nearest_indices: list[int] | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    """Return the reference pair used for visualization and metric interpretation."""

    if reference_inputs is None or nearest_indices is None:
        return None, None, "exact_target"
    index_tensor = torch.tensor(nearest_indices, dtype=torch.long)
    reference_x = reference_inputs.detach().cpu().index_select(0, index_tensor)
    reference_y = None if reference_targets is None else reference_targets.detach().cpu().index_select(0, index_tensor)
    return reference_x, reference_y, "nearest_client_train"


def _evaluate_reconstruction(
    name: str,
    reconstructed_x: torch.Tensor,
    real_x: torch.Tensor,
    reconstructed_y: torch.Tensor | None,
    real_y: torch.Tensor,
    iterations: int,
    elapsed: float,
    threshold: float,
    objective_mse: float,
    data_range: float,
    ssim_threshold: float | None,
    target_type: str,
    reference_inputs: torch.Tensor | None = None,
    reference_targets: torch.Tensor | None = None,
    reference_metric: str = "reconstruction_mse",
    report_metrics: set[str] | None = None,
) -> AttackResult:
    """Build an attack result from a reconstructed input."""

    exact_target_value = torch.mean((reconstructed_x.detach().cpu() - real_x.detach().cpu()) ** 2).item()
    psnr = _compute_psnr(exact_target_value, data_range)
    ssim = _compute_ssim(reconstructed_x, real_x, data_range)
    nearest_client_train_mse = None
    nearest_client_train_indices = None
    if reference_inputs is not None:
        nearest_client_train_mse, nearest_client_train_indices = _nearest_reference_mse(reconstructed_x, reference_inputs)
    metric_name = reference_metric
    primary_mse = float(exact_target_value)
    reference_x = real_x.detach().cpu().clone()
    reference_y = real_y.detach().cpu().clone()
    reference_label = "exact_target"
    if metric_name == "nearest_client_train_mse":
        if nearest_client_train_mse is None:
            metric_name = "reconstruction_mse"
        else:
            primary_mse = float(nearest_client_train_mse)
            selected_reference_x, selected_reference_y, reference_label = _select_reference_targets(
                reference_inputs,
                reference_targets,
                nearest_client_train_indices,
            )
            if selected_reference_x is not None:
                reference_x = selected_reference_x
            if selected_reference_y is not None:
                reference_y = selected_reference_y
    success = primary_mse <= threshold
    if metric_name == "reconstruction_mse" and ssim_threshold is not None:
        success = success or ssim >= ssim_threshold
    report_metrics = set() if report_metrics is None else set(report_metrics)
    expose_exact_target = metric_name == "reconstruction_mse" or "exact_target_mse" in report_metrics
    expose_nearest = metric_name == "nearest_client_train_mse" or "nearest_client_train_mse" in report_metrics
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
        exact_target_mse=float(exact_target_value) if expose_exact_target else None,
        nearest_client_train_mse=None if (nearest_client_train_mse is None or not expose_nearest) else float(nearest_client_train_mse),
        nearest_client_train_indices=nearest_client_train_indices,
        metric_name=metric_name,
        real_x=real_x.detach().cpu().clone(),
        real_y=real_y.detach().cpu().clone(),
        reference_x=reference_x,
        reference_y=reference_y,
        reference_label=reference_label,
        reconstructed_x=reconstructed_x.detach().cpu().clone(),
        reconstructed_y=None if reconstructed_y is None else reconstructed_y.detach().cpu().clone(),
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
    reference_targets: torch.Tensor | None = None,
) -> AttackResult:
    """Run one configurable reconstruction attack loop."""

    resolved_target_type = _normalize_target_type(target_type, config)
    resolved_reference_metric = _normalize_reference_metric(config, resolved_target_type)
    report_metrics = _normalize_report_metrics(config, resolved_target_type)
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
    best_y = real_y.to(device)

    for restart in range(restarts):
        with _attack_rng_context(device):
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
            restart_best_y = dummy_y.detach().clone()

            for _ in range(steps):
                if optimizer_name == "lbfgs":
                    holder: dict[str, float] = {}

                    def closure() -> torch.Tensor:
                        """Evaluate the current dummy variables for one LBFGS step."""

                        optimizer.zero_grad(set_to_none=True)
                        if resolved_target_type == "gradient":
                            dist = _gradient_distance(model, dummy_x, dummy_y, prepared_target, config)
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
                        dist = _gradient_distance(model, dummy_x, dummy_y, prepared_target, config)
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
                    restart_best_y = dummy_y.detach().clone()
        if restart_best_objective < best_objective_mse:
            best_objective_mse = restart_best_objective
            best_x = restart_best_x
            best_y = restart_best_y
    elapsed = time.perf_counter() - overall_start
    return _evaluate_reconstruction(
        name,
        best_x,
        real_x,
        best_y,
        real_y,
        steps,
        elapsed,
        threshold,
        best_objective_mse,
        data_range,
        ssim_threshold,
        resolved_target_type,
        reference_inputs=reference_inputs,
        reference_metric=resolved_reference_metric,
        report_metrics=report_metrics,
    )



def attach_attack_metadata(
    result: AttackResult,
    *,
    client_id: str,
    round_index: int,
    sample_index: int,
) -> AttackResult:
    """Attach immutable sample metadata to one attack result."""

    result.client_id = client_id
    result.round_index = round_index
    result.sample_index = sample_index
    return result


def save_attack_artifacts(output_dir: Path, results: list[AttackResult]) -> list[dict[str, Any]]:
    """Persist attack reconstruction tensors and return JSON-ready records."""

    artifact_root = output_dir / "attack_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        round_index = 0 if result.round_index is None else int(result.round_index)
        sample_index = 0 if result.sample_index is None else int(result.sample_index)
        client_id = result.client_id or "unknown_client"
        filename = f"{result.name.lower()}_{index:05d}.pt"
        relative_path = Path(f"round_{round_index:04d}") / client_id / f"sample_{sample_index:04d}" / filename
        artifact_path = artifact_root / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": result.name,
                "client_id": result.client_id,
                "round_index": result.round_index,
                "sample_index": result.sample_index,
                "target_type": result.target_type,
                "primary_metric_name": result.metric_name,
                "primary_metric_value": result.mse,
                "real_x": None if result.real_x is None else result.real_x.detach().cpu(),
                "real_y": None if result.real_y is None else result.real_y.detach().cpu(),
                "reference_x": None if result.reference_x is None else result.reference_x.detach().cpu(),
                "reference_y": None if result.reference_y is None else result.reference_y.detach().cpu(),
                "reference_label": result.reference_label,
                "reconstructed_x": None if result.reconstructed_x is None else result.reconstructed_x.detach().cpu(),
                "reconstructed_y": None if result.reconstructed_y is None else result.reconstructed_y.detach().cpu(),
                "plot_real_x": None if getattr(result, "plot_real_x", None) is None else getattr(result, "plot_real_x").detach().cpu(),
                "plot_real_y": None if getattr(result, "plot_real_y", None) is None else getattr(result, "plot_real_y").detach().cpu(),
                "plot_reference_x": None if getattr(result, "plot_reference_x", None) is None else getattr(result, "plot_reference_x").detach().cpu(),
                "plot_reference_y": None if getattr(result, "plot_reference_y", None) is None else getattr(result, "plot_reference_y").detach().cpu(),
                "plot_reconstructed_x": None if getattr(result, "plot_reconstructed_x", None) is None else getattr(result, "plot_reconstructed_x").detach().cpu(),
                "plot_reconstructed_y": None if getattr(result, "plot_reconstructed_y", None) is None else getattr(result, "plot_reconstructed_y").detach().cpu(),
                "exact_target_mse": result.exact_target_mse,
                "nearest_client_train_mse": result.nearest_client_train_mse,
                "nearest_client_train_indices": result.nearest_client_train_indices,
            },
            artifact_path,
        )
        result.artifact_path = str(Path("attack_artifacts") / relative_path)
        records.append(result.to_record())
    return records

def dlg_attack(
    config: dict[str, Any],
    state: StateDict,
    target: list[torch.Tensor] | StateDict,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    device: torch.device,
    target_type: str | None = None,
    reference_inputs: torch.Tensor | None = None,
    reference_targets: torch.Tensor | None = None,
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
        reference_targets=reference_targets,
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
    reference_targets: torch.Tensor | None = None,
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
        reference_targets=reference_targets,
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


def _summarize_attack_subset(
    subset: list[AttackResult],
    *,
    primary_metric_name: str,
    success_rate_threshold: float,
) -> dict[str, Any]:
    """Return one aggregated attack summary for an arbitrary filtered subset."""

    methods: dict[str, dict[str, float | int | bool | None | str]] = {}
    for name in sorted({result.name for result in subset}):
        method_subset = [result for result in subset if result.name == name]
        total = len(method_subset)
        success_count = sum(result.success for result in method_subset)
        avg_primary_metric_value = _mean_finite([result.mse for result in method_subset])
        best_primary_metric_value = _mean_finite(sorted([result.mse for result in method_subset])[:1])
        avg_objective_mse = _mean_finite([result.gradient_mse for result in method_subset])
        method_record: dict[str, Any] = {
            "primary_metric_name": method_subset[0].metric_name if method_subset else primary_metric_name,
            "target_type": method_subset[0].target_type if method_subset else None,
            "success_count": success_count,
            "total_count": total,
            "success_rate": success_count / total if total else 0.0,
            "success_rate_percent": round((success_count / total if total else 0.0) * 100.0, 2),
            "avg_primary_metric_value": avg_primary_metric_value,
            "best_primary_metric_value": best_primary_metric_value,
            "avg_psnr": _mean_finite([result.psnr for result in method_subset]),
            "avg_ssim": _mean_finite([result.ssim for result in method_subset]),
            "best_ssim": _mean_finite(sorted([result.ssim for result in method_subset], reverse=True)[:1]),
            "avg_objective_mse": avg_objective_mse,
            "avg_time_seconds": _mean_finite([result.time_seconds for result in method_subset]),
            "passes": (success_count / total if total else 0.0) <= success_rate_threshold,
        }
        exact_values = [result.exact_target_mse for result in method_subset if result.exact_target_mse is not None]
        nearest_values = [result.nearest_client_train_mse for result in method_subset if result.nearest_client_train_mse is not None]
        if exact_values:
            method_record["avg_exact_target_mse"] = _mean_finite(exact_values)
        if nearest_values:
            method_record["avg_nearest_client_train_mse"] = _mean_finite(nearest_values)
        methods[name] = method_record

    overall_success_rate = attack_success_rate(subset)
    overall_avg_primary_metric_value = _mean_finite([result.mse for result in subset])
    overall_best_primary_metric_value = _mean_finite(sorted([result.mse for result in subset])[:1])
    overall_record: dict[str, Any] = {
        "primary_metric_name": primary_metric_name,
        "primary_metric_direction": "higher_is_more_private",
        "target_type": subset[0].target_type if subset else None,
        "success_rate_threshold": success_rate_threshold,
        "overall_avg_primary_metric_value": overall_avg_primary_metric_value,
        "overall_best_primary_metric_value": overall_best_primary_metric_value,
        "overall_avg_psnr": _mean_finite([result.psnr for result in subset]),
        "overall_avg_ssim": _mean_finite([result.ssim for result in subset]),
        "overall_avg_objective_mse": _mean_finite([result.gradient_mse for result in subset]),
        "overall_success_rate": overall_success_rate,
        "overall_success_rate_percent": round(overall_success_rate * 100.0, 2),
        "overall_passes": overall_success_rate <= success_rate_threshold,
        "methods": methods,
    }
    overall_exact_values = [result.exact_target_mse for result in subset if result.exact_target_mse is not None]
    overall_nearest_values = [result.nearest_client_train_mse for result in subset if result.nearest_client_train_mse is not None]
    if overall_exact_values:
        overall_record["overall_avg_exact_target_mse"] = _mean_finite(overall_exact_values)
    if overall_nearest_values:
        overall_record["overall_avg_nearest_client_train_mse"] = _mean_finite(overall_nearest_values)
    return overall_record


def summarize_attack_results(results: list[AttackResult], success_rate_threshold: float = 0.03) -> dict[str, Any]:
    """Summarize DLG/iDLG metrics with both merged and per-client views."""

    primary_metric_name = results[0].metric_name if results else "reconstruction_mse"
    summary = _summarize_attack_subset(
        results,
        primary_metric_name=primary_metric_name,
        success_rate_threshold=success_rate_threshold,
    )
    clients: dict[str, dict[str, Any]] = {}
    for client_id in sorted({str(result.client_id) for result in results if result.client_id is not None}):
        client_subset = [result for result in results if result.client_id == client_id]
        clients[client_id] = _summarize_attack_subset(
            client_subset,
            primary_metric_name=primary_metric_name,
            success_rate_threshold=success_rate_threshold,
        )
    summary["clients"] = clients
    return summary
