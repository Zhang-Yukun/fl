"""Shared runtime helpers for device resolution, thread limits, and seeding."""

from __future__ import annotations

import os
import random
from typing import Any

import torch
from loguru import logger

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in the runtime env
    np = None

from fedlab.utils.random import seed_cuda_device


def configure_torch_runtime(config: dict[str, Any]) -> None:
    """Apply CPU thread limits from runtime config before workload start."""

    runtime_cfg = config.get("runtime", {})
    num_threads = runtime_cfg.get("num_threads")
    interop_threads = runtime_cfg.get("num_interop_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
        logger.info("Set torch num_threads={}", int(num_threads))
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(interop_threads))
            logger.info("Set torch num_interop_threads={}", int(interop_threads))
        except RuntimeError as exc:
            logger.warning("Could not set torch interop threads after runtime start: {}", exc)


def setup_seed(seed: int, deterministic: bool = True, *, device: torch.device | None = None) -> None:
    """Set Python, NumPy, and torch random sources to a reproducible state."""

    seed_value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed_value)
    if np is not None:
        np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    seed_cuda_device(seed_value, device)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception as exc:  # pragma: no cover - depends on torch build/runtime
        logger.warning("Could not set deterministic torch algorithms: {}", exc)
    logger.info("Set runtime seed={} deterministic={} device={}", seed_value, deterministic, device or "cpu")


def resolve_device(config: dict[str, Any]) -> torch.device:
    """Resolve the configured torch device with a CPU fallback."""

    requested = str(config.get("runtime", {}).get("device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    return torch.device(requested)


def configure_random_seed(config: dict[str, Any], *, device: torch.device | None = None) -> None:
    """Apply the configured runtime seed when one is provided."""

    runtime_cfg = config.get("runtime", {})
    seed = runtime_cfg.get("seed")
    if seed is None:
        return
    resolved_device = resolve_device(config) if device is None else device
    setup_seed(int(seed), deterministic=bool(runtime_cfg.get("deterministic", True)), device=resolved_device)
