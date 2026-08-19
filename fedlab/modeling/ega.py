"""Encoded Gradient Aggregation (EGA) building blocks and synthetic pretraining.

Example:
    ``codec = load_ega_codec(config, device=torch.device("cpu"), num_clients=3, allow_pretrain=True)``
    loads an existing EGA encoder-decoder or pretrains one on synthetic integer
    vectors following the paper objective.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedlab.utils.serialization import StateDict


@dataclass
class EncodedStatePayload:
    """Encoded model update payload uploaded by one client in EGA.

    The payload stores encoded block vectors, optional transmission-side
    quantization metadata, and the scalar normalization used by stochastic
    integer quantization.
    """

    names: list[str]
    shapes: list[tuple[int, ...]]
    encoded_blocks: torch.Tensor
    original_numel: int
    block_size: int
    encoded_dim: int
    quantization_level: int
    normalization: float
    contribution_scale: float
    encoded_scale: float | None = None
    encoded_dtype: str = "float32"
    observed_update_absmax: float = 0.0

    @property
    def nbytes(self) -> int:
        """Return the numeric payload size in bytes."""

        scale_bytes = 4 if self.encoded_scale is not None else 0
        return self.encoded_blocks.numel() * self.encoded_blocks.element_size() + scale_bytes

    @property
    def num_parameters(self) -> int:
        """Return the number of transmitted scalar values."""

        return int(self.encoded_blocks.numel()) + (1 if self.encoded_scale is not None else 0)


class ResidualMLPBlock(nn.Module):
    """Simple residual MLP block used by the EGA encoder and decoder."""

    def __init__(self, dim: int):
        """Build one residual block with a fixed hidden width."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one residual MLP update."""

        return self.activation(x + self.net(x))


class EgaAutoEncoder(nn.Module):
    """Symmetric residual MLP encoder-decoder used by EGA."""

    def __init__(
        self,
        block_size: int,
        encoded_dim: int,
        hidden_dim: int,
        residual_blocks: int = 2,
    ) -> None:
        """Initialize encoder and decoder widths for one EGA codec."""

        super().__init__()
        self.block_size = int(block_size)
        self.encoded_dim = int(encoded_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_blocks = int(residual_blocks)
        self.encoder = self._build_mlp(self.block_size, self.encoded_dim)
        self.decoder = self._build_mlp(self.encoded_dim, self.block_size)

    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """Build one residual MLP stack for the encoder or decoder side."""

        layers: list[nn.Module] = [nn.Linear(input_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.residual_blocks):
            layers.append(ResidualMLPBlock(self.hidden_dim))
        layers.append(nn.Linear(self.hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def encode_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """Encode one batch of quantized blocks."""

        return self.encoder(blocks.to(torch.float32))

    def decode_blocks(self, encoded_blocks: torch.Tensor) -> torch.Tensor:
        """Decode one batch of aggregated encoded blocks."""

        return self.decoder(encoded_blocks.to(torch.float32))

    def forward(self, grouped_blocks: torch.Tensor) -> torch.Tensor:
        """Decode the mean aggregation of one grouped batch.

        ``grouped_blocks`` has shape ``[batch, num_clients, block_size]`` and the
        output has shape ``[batch, block_size]``.
        """

        encoded = self.encode_blocks(grouped_blocks.reshape(-1, grouped_blocks.shape[-1]))
        encoded = encoded.reshape(grouped_blocks.shape[0], grouped_blocks.shape[1], self.encoded_dim)
        aggregated = encoded.mean(dim=1)
        return self.decode_blocks(aggregated)


def _codec_device(codec: EgaAutoEncoder) -> torch.device:
    """Return the device that owns one EGA codec instance."""

    return next(codec.parameters()).device


def _ega_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the EGA-specific config subtree."""

    return config.get("ega", {})


def _flatten_state(update: StateDict) -> tuple[list[str], list[tuple[int, ...]], torch.Tensor]:
    """Flatten a state dict while preserving names and shapes."""

    names = list(update.keys())
    shapes = [tuple(update[name].shape) for name in names]
    flat = torch.cat([update[name].reshape(-1).detach().cpu().to(torch.float32) for name in names])
    return names, shapes, flat


def _unflatten_state(
    names: list[str],
    shapes: list[tuple[int, ...]],
    flat: torch.Tensor,
    original_numel: int,
) -> StateDict:
    """Restore a flat vector into the original named state layout."""

    trimmed = flat[:original_numel]
    result: StateDict = OrderedDict()
    offset = 0
    for name, shape in zip(names, shapes):
        numel = int(math.prod(shape)) if shape else 1
        result[name] = trimmed[offset : offset + numel].reshape(shape).detach().cpu().to(torch.float32)
        offset += numel
    return result


def pack_flat_blocks(flat: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    """Pack a flat vector into equally-sized blocks with zero padding."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)], dim=0)
    return flat.reshape(-1, block_size), pad


def unpack_flat_blocks(blocks: torch.Tensor, original_numel: int) -> torch.Tensor:
    """Flatten decoded blocks and remove zero padding."""

    return blocks.reshape(-1)[:original_numel]


def stochastic_quantize_block_vector(
    vector: torch.Tensor,
    *,
    quantization_level: int,
    normalization: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Quantize a real vector into the bounded integer domain ``[-s, s]``.

    This follows the paper's stochastic rounding rule after scaling by
    ``s / n``. Values outside the nominal range are clipped.
    """

    if quantization_level <= 0:
        raise ValueError("quantization_level must be positive")
    if normalization <= 0:
        raise ValueError("normalization must be positive")
    scaled = torch.clamp(
        vector.to(torch.float32) * (float(quantization_level) / float(normalization)),
        -float(quantization_level),
        float(quantization_level),
    )
    lower = torch.floor(scaled)
    probability = scaled - lower
    if generator is None:
        random = torch.rand(scaled.shape, dtype=torch.float32)
    else:
        random = torch.rand(scaled.shape, generator=generator, dtype=torch.float32)
    quantized = lower + (random < probability).to(torch.float32)
    return torch.clamp(quantized, -float(quantization_level), float(quantization_level))


def dequantize_block_vector(
    quantized: torch.Tensor,
    *,
    quantization_level: int,
    normalization: float,
) -> torch.Tensor:
    """Map a quantized integer vector back to the real domain."""

    if quantization_level <= 0:
        raise ValueError("quantization_level must be positive")
    return quantized.to(torch.float32) * (float(normalization) / float(quantization_level))


def quantize_encoded_blocks(
    encoded_blocks: torch.Tensor,
    *,
    dtype: str = "float32",
    stochastic_rounding: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, float | None, str]:
    """Quantize encoded blocks before transmission."""

    normalized_dtype = str(dtype).lower()
    if normalized_dtype in {"float32", "fp32"}:
        return encoded_blocks.detach().cpu().to(torch.float32), None, "float32"
    if normalized_dtype in {"float16", "fp16"}:
        return encoded_blocks.detach().cpu().to(torch.float16), None, "float16"
    if normalized_dtype in {"bfloat16", "bf16"}:
        return encoded_blocks.detach().cpu().to(torch.bfloat16), None, "bfloat16"
    if normalized_dtype not in {"int8", "qint8", "absmax_int8", "scaled_int8"}:
        raise ValueError(f"Unsupported EGA encoded dtype: {dtype}")
    base = encoded_blocks.detach().cpu().to(torch.float32)
    max_abs = float(base.abs().max().item())
    scale = max(max_abs / 127.0, 1e-12)
    normalized = torch.clamp(base / scale, -127.0, 127.0)
    if stochastic_rounding:
        lower = torch.floor(normalized)
        probability = normalized - lower
        if generator is None:
            random = torch.rand(normalized.shape, dtype=torch.float32)
        else:
            random = torch.rand(normalized.shape, generator=generator, dtype=torch.float32)
        rounded = lower + (random < probability).to(torch.float32)
    else:
        rounded = torch.round(normalized)
    quantized = torch.clamp(rounded, -127.0, 127.0).to(torch.int8)
    return quantized, scale, "int8"


def dequantize_encoded_blocks(payload: EncodedStatePayload) -> torch.Tensor:
    """Restore one transmitted encoded payload to float32 blocks."""

    if payload.encoded_scale is None:
        return payload.encoded_blocks.detach().cpu().to(torch.float32)
    return payload.encoded_blocks.detach().cpu().to(torch.float32) * float(payload.encoded_scale)


def encode_state_update(
    update: StateDict,
    codec: EgaAutoEncoder,
    *,
    quantization_level: int,
    normalization: float,
    block_size: int,
    contribution_scale: float,
    generator: torch.Generator | None = None,
    encoded_dtype: str = "float32",
    encoded_stochastic_rounding: bool = False,
    encoded_noise_std: float = 0.0,
) -> EncodedStatePayload:
    """Encode one dense state update into the EGA upload domain."""

    names, shapes, flat = _flatten_state(update)
    scaled = flat * float(contribution_scale)
    blocks, _ = pack_flat_blocks(scaled, block_size)
    quantized = stochastic_quantize_block_vector(
        blocks,
        quantization_level=quantization_level,
        normalization=normalization,
        generator=generator,
    )
    codec_device = _codec_device(codec)
    encoded = codec.encode_blocks(quantized.to(codec_device)).detach().cpu().to(torch.float32)
    if encoded_noise_std > 0:
        if generator is None:
            noise = torch.randn(encoded.shape, dtype=torch.float32) * float(encoded_noise_std)
        else:
            noise = torch.randn(encoded.shape, generator=generator, dtype=torch.float32) * float(encoded_noise_std)
        encoded = encoded + noise
    transmitted_encoded, encoded_scale, resolved_encoded_dtype = quantize_encoded_blocks(
        encoded,
        dtype=encoded_dtype,
        stochastic_rounding=encoded_stochastic_rounding,
        generator=generator,
    )
    return EncodedStatePayload(
        names=names,
        shapes=shapes,
        encoded_blocks=transmitted_encoded,
        original_numel=flat.numel(),
        block_size=int(block_size),
        encoded_dim=int(codec.encoded_dim),
        quantization_level=int(quantization_level),
        normalization=float(normalization),
        contribution_scale=float(contribution_scale),
        encoded_scale=encoded_scale,
        encoded_dtype=resolved_encoded_dtype,
        observed_update_absmax=float(scaled.abs().max().item()) if scaled.numel() else 0.0,
    )


def decode_mean_encoded_payload(
    payloads: list[EncodedStatePayload],
    codec: EgaAutoEncoder,
) -> StateDict:
    """Decode the mean aggregation of EGA payloads back to a dense update."""

    if not payloads:
        raise ValueError("payloads must be non-empty")
    first = payloads[0]
    codec_device = _codec_device(codec)
    aggregated = torch.stack([dequantize_encoded_blocks(payload) for payload in payloads], dim=0).mean(dim=0).to(codec_device)
    decoded = codec.decode_blocks(aggregated).detach().cpu()
    flat = unpack_flat_blocks(
        dequantize_block_vector(
            decoded,
            quantization_level=first.quantization_level,
            normalization=first.normalization,
        ),
        original_numel=first.original_numel,
    )
    return _unflatten_state(first.names, first.shapes, flat, first.original_numel)


def decode_attack_view_from_mean_difference(
    payloads: list[EncodedStatePayload],
    target_index: int,
    codec: EgaAutoEncoder,
) -> StateDict:
    """Reconstruct one client's contribution from the server-side EGA attack view."""

    if not payloads:
        raise ValueError("payloads must be non-empty")
    first = payloads[0]
    codec_device = _codec_device(codec)
    encoded_stack = torch.stack([dequantize_encoded_blocks(payload) for payload in payloads], dim=0)
    zero_blocks = codec.encode_blocks(
        torch.zeros((encoded_stack.shape[1], first.block_size), dtype=torch.float32, device=codec_device)
    ).detach().cpu()
    mean_all = encoded_stack.mean(dim=0).to(codec_device)
    modified = encoded_stack.clone()
    modified[target_index] = zero_blocks
    mean_without_target = modified.mean(dim=0).to(codec_device)
    decoded = codec.decode_blocks(mean_all).detach().cpu() - codec.decode_blocks(mean_without_target).detach().cpu()
    flat = unpack_flat_blocks(
        dequantize_block_vector(
            decoded,
            quantization_level=first.quantization_level,
            normalization=first.normalization,
        ),
        original_numel=first.original_numel,
    )
    return _unflatten_state(first.names, first.shapes, flat, first.original_numel)


def build_ega_model(config: dict[str, Any]) -> EgaAutoEncoder:
    """Build an EGA encoder-decoder from config."""

    ega_cfg = _ega_config(config)
    block_size = int(ega_cfg.get("block_size", 256))
    encoded_dim = int(ega_cfg.get("encoded_dim", block_size))
    hidden_dim = int(ega_cfg.get("hidden_dim", max(block_size, encoded_dim) * 2))
    residual_blocks = int(ega_cfg.get("residual_blocks", 2))
    return EgaAutoEncoder(
        block_size=block_size,
        encoded_dim=encoded_dim,
        hidden_dim=hidden_dim,
        residual_blocks=residual_blocks,
    )


def resolve_ega_artifact_path(config: dict[str, Any], num_clients: int) -> Path:
    """Resolve the checkpoint path used to persist the pretrained EGA codec."""

    ega_cfg = _ega_config(config)
    configured = ega_cfg.get("artifact_path")
    if configured:
        return Path(configured)
    experiment_name = str(config.get("experiment", {}).get("name", "ega"))
    return Path("artifacts") / "ega" / f"{experiment_name}_m{num_clients}_b{int(ega_cfg.get('block_size', 256))}_h{int(ega_cfg.get('encoded_dim', int(ega_cfg.get('block_size', 256))))}_s{int(ega_cfg.get('quantization_level', 64))}.pt"


def _build_synthetic_ega_loader(
    *,
    groups: int,
    num_clients: int,
    block_size: int,
    quantization_level: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Build a synthetic integer-domain dataset for EGA pretraining."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    data = torch.randint(
        low=-int(quantization_level),
        high=int(quantization_level) + 1,
        size=(int(groups), int(num_clients), int(block_size)),
        generator=generator,
        dtype=torch.int32,
    ).to(torch.float32)
    return DataLoader(TensorDataset(data), batch_size=int(batch_size), shuffle=True)


def pretrain_ega_codec(
    config: dict[str, Any],
    *,
    num_clients: int,
    device: torch.device,
    output_path: Path,
) -> Path:
    """Offline-pretrain the EGA codec on synthetic quantized vectors."""

    ega_cfg = _ega_config(config)
    pretrain_cfg = ega_cfg.get("pretrain", {})
    codec = build_ega_model(config).to(device)
    block_size = int(ega_cfg.get("block_size", 256))
    quantization_level = int(ega_cfg.get("quantization_level", 64))
    train_loader = _build_synthetic_ega_loader(
        groups=int(pretrain_cfg.get("train_groups", 1024)),
        num_clients=num_clients,
        block_size=block_size,
        quantization_level=quantization_level,
        batch_size=int(pretrain_cfg.get("batch_size", 32)),
        seed=int(pretrain_cfg.get("seed", config.get("runtime", {}).get("seed", 2026))),
    )
    val_loader = _build_synthetic_ega_loader(
        groups=int(pretrain_cfg.get("val_groups", 256)),
        num_clients=num_clients,
        block_size=block_size,
        quantization_level=quantization_level,
        batch_size=int(pretrain_cfg.get("batch_size", 32)),
        seed=int(pretrain_cfg.get("seed", config.get("runtime", {}).get("seed", 2026))) + 17,
    )
    optimizer = torch.optim.Adam(codec.parameters(), lr=float(pretrain_cfg.get("lr", 1e-3)))
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    completed_epochs = 0
    patience = pretrain_cfg.get("patience")
    patience = None if patience is None else int(patience)
    min_delta = float(pretrain_cfg.get("min_delta", 0.0))
    bad_epochs = 0
    epochs = int(pretrain_cfg.get("epochs", 100))
    for epoch in range(epochs):
        codec.train()
        train_loss = 0.0
        train_count = 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            target = batch.mean(dim=1)
            prediction = codec(batch)
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * batch.shape[0]
            train_count += int(batch.shape[0])
        codec.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                prediction = codec(batch)
                target = batch.mean(dim=1)
                loss = criterion(prediction, target)
                val_loss += float(loss.item()) * batch.shape[0]
                val_count += int(batch.shape[0])
        avg_train = train_loss / max(train_count, 1)
        avg_val = val_loss / max(val_count, 1)
        logger.info(
            "EGA pretrain epoch {} train_loss={:.6f} val_loss={:.6f}",
            epoch,
            avg_train,
            avg_val,
        )
        completed_epochs = epoch + 1
        if avg_val < (best_val - min_delta):
            best_val = avg_val
            best_epoch = epoch
            bad_epochs = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in codec.state_dict().items()}
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                logger.info(
                    "EGA pretrain early stopping at epoch {} best_epoch={} best_val_loss={:.6f}",
                    epoch,
                    best_epoch,
                    best_val,
                )
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state if best_state is not None else codec.state_dict(),
            "config": {
                "block_size": block_size,
                "encoded_dim": codec.encoded_dim,
                "hidden_dim": codec.hidden_dim,
                "residual_blocks": codec.residual_blocks,
                "quantization_level": quantization_level,
                "num_clients": num_clients,
            },
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            "stopped_early": completed_epochs < epochs,
        },
        output_path,
    )
    logger.info(
        "Saved pretrained EGA codec to {} best_val_loss={:.6f} best_epoch={} completed_epochs={}",
        output_path,
        best_val,
        best_epoch,
        completed_epochs,
    )
    return output_path


def load_ega_codec(
    config: dict[str, Any],
    *,
    device: torch.device,
    num_clients: int,
    allow_pretrain: bool,
) -> EgaAutoEncoder:
    """Load a pretrained EGA codec, optionally creating it on demand."""

    path = resolve_ega_artifact_path(config, num_clients)
    ega_cfg = _ega_config(config)
    block_size = int(ega_cfg.get("block_size", 256))
    encoded_dim = int(ega_cfg.get("encoded_dim", block_size))
    hidden_dim = int(ega_cfg.get("hidden_dim", max(block_size, encoded_dim) * 2))
    expected_spec = {
        "block_size": block_size,
        "encoded_dim": encoded_dim,
        "hidden_dim": hidden_dim,
        "residual_blocks": int(ega_cfg.get("residual_blocks", 2)),
        "quantization_level": int(ega_cfg.get("quantization_level", 64)),
        "num_clients": int(num_clients),
    }
    if not path.exists():
        if not allow_pretrain:
            raise FileNotFoundError(f"EGA codec artifact not found: {path}")
        logger.info("EGA codec artifact missing at {}; start synthetic pretraining", path)
        requested_device = ega_cfg.get("pretrain", {}).get(
            "device",
            config.get("runtime", {}).get("device", str(device)),
        )
        pretrain_device = torch.device(str(requested_device))
        if pretrain_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested EGA pretrain device {} is unavailable; falling back to cpu", pretrain_device)
            pretrain_device = torch.device("cpu")
        pretrain_ega_codec(config, num_clients=num_clients, device=pretrain_device, output_path=path)
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_spec = checkpoint.get("config", {})
    if any(checkpoint_spec.get(key) != value for key, value in expected_spec.items()):
        if not allow_pretrain:
            raise RuntimeError(f"EGA codec artifact at {path} does not match current config")
        logger.info("EGA codec artifact at {} does not match current config; retraining", path)
        requested_device = ega_cfg.get("pretrain", {}).get(
            "device",
            config.get("runtime", {}).get("device", str(device)),
        )
        pretrain_device = torch.device(str(requested_device))
        if pretrain_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("Requested EGA pretrain device {} is unavailable; falling back to cpu", pretrain_device)
            pretrain_device = torch.device("cpu")
        pretrain_ega_codec(config, num_clients=num_clients, device=pretrain_device, output_path=path)
        checkpoint = torch.load(path, map_location="cpu")
    codec = build_ega_model(config)
    codec.load_state_dict(checkpoint["state_dict"])
    codec.to(device)
    codec.eval()
    logger.info("Loaded EGA codec from {}", path)
    return codec
