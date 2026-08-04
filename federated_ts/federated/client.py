"""Federated client implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.peft import is_fedpetuning, serialize_trainable_state, subset_state
from federated_ts.utils.serialization import (
    SparseUpdate,
    StateDict,
    compress_randomk,
    compress_topk,
    dequantize_state_update,
    quantize_state_update,
    load_serialized,
    privatize_state_update,
    serialize_model,
    state_num_bytes,
    state_num_parameters,
    subtract_state,
)
from federated_ts.engine.training import first_batch_gradient, first_batch_sample, train_one_epoch


@dataclass
class ClientResult:
    """Payload and communication metadata returned by one client update."""

    client_id: str
    num_samples: int
    loss: float
    state: StateDict | None = None
    sparse_update: SparseUpdate | None = None
    dense_bytes: int = 0
    dense_parameters: int = 0
    download_bytes: int = 0
    download_parameters: int = 0
    dense_download_reference_bytes: int = 0
    dense_download_reference_parameters: int = 0
    upload_bytes: int = 0
    upload_parameters: int = 0
    payload_kind: str = "dense_update"
    compressor: str = "none"
    privacy_clip_norm: float = 0.0
    privacy_noise_multiplier: float = 0.0

    @property
    def sent_bytes(self) -> int:
        """Backward-compatible alias for upload bytes."""

        return self.upload_bytes


class FederatedClient:
    """Local trainer that receives global parameters and returns an update."""

    def __init__(self, client_id: str, train_loader, config: dict[str, Any], device: torch.device):
        """Create a client bound to one local training loader."""

        self.client_id = client_id
        self.train_loader = train_loader
        self.config = config
        self.device = device

    def _upload_quantization_generator(self, round_index: int) -> torch.Generator | None:
        """Create a deterministic per-client generator for randomized upload quantization."""

        seed = self.config.get("federated", {}).get("quantization_seed")
        if seed is None:
            return None
        offset = sum(ord(char) for char in self.client_id)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + round_index * 1000 + offset)
        return generator

    def train(self, global_state: StateDict, compressed: bool = False, round_index: int = 0) -> ClientResult:
        """Train locally from global weights and return the transmitted payload.

        Example:
            Standard FedAvg-style clients upload dense model updates
            ``local_state - received_global_state`` instead of full model states.
        """

        algorithm = str(self.config["federated"].get("algorithm", "fedavg")).lower()
        received_global_state = global_state
        download_state = global_state
        if algorithm == "secure_quantized_fedavg":
            quantization_dtype = str(self.config.get("federated", {}).get("quantization_dtype", "float16"))
            download_state = quantize_state_update(global_state, dtype=quantization_dtype)
            received_global_state = dequantize_state_update(download_state)

        model = build_model(self.config).to(self.device)
        load_serialized(model, received_global_state, self.device)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=float(self.config["training"].get("lr", 1e-3)))
        losses = []
        for _ in range(int(self.config["federated"].get("local_epochs", 1))):
            losses.append(train_one_epoch(model, self.train_loader, optimizer, self.device))
        local_state = serialize_model(model)
        dense_bytes = state_num_bytes(local_state)
        dense_parameters = state_num_parameters(local_state)
        if is_fedpetuning(self.config):
            trainable_state = serialize_trainable_state(model)
            global_trainable_state = subset_state(global_state, trainable_state.keys())
            return ClientResult(
                client_id=self.client_id,
                num_samples=len(self.train_loader.dataset),
                loss=float(sum(losses) / len(losses)),
                state=trainable_state,
                dense_bytes=dense_bytes,
                dense_parameters=dense_parameters,
                download_bytes=state_num_bytes(global_trainable_state),
                download_parameters=state_num_parameters(global_trainable_state),
                dense_download_reference_bytes=state_num_bytes(global_trainable_state),
                dense_download_reference_parameters=state_num_parameters(global_trainable_state),
                upload_bytes=state_num_bytes(trainable_state),
                upload_parameters=state_num_parameters(trainable_state),
                payload_kind="fedpetuning_trainable_state",
                compressor="trainable_subset",
            )
        common = dict(
            client_id=self.client_id,
            num_samples=len(self.train_loader.dataset),
            loss=float(sum(losses) / len(losses)),
            dense_bytes=dense_bytes,
            dense_parameters=dense_parameters,
            download_bytes=state_num_bytes(download_state),
            download_parameters=state_num_parameters(download_state),
            dense_download_reference_bytes=state_num_bytes(global_state),
            dense_download_reference_parameters=state_num_parameters(global_state),
        )
        if algorithm == "secure_quantized_fedavg":
            update = subtract_state(local_state, received_global_state)
            privacy_cfg = self.config.get("privacy", {})
            privacy_clip_norm = float(privacy_cfg.get("clip_norm", 0.0))
            privacy_noise_multiplier = float(privacy_cfg.get("noise_multiplier", 0.0))
            update = privatize_state_update(update, privacy_clip_norm, privacy_noise_multiplier)
            quantized = quantize_state_update(
                update,
                dtype=str(self.config.get("federated", {}).get("quantization_dtype", "float16")),
                stochastic_rounding=bool(self.config.get("federated", {}).get("quantization_stochastic_rounding", False)),
                generator=self._upload_quantization_generator(round_index),
            )
            return ClientResult(
                **common,
                state=quantized,
                upload_bytes=state_num_bytes(quantized),
                upload_parameters=state_num_parameters(quantized),
                payload_kind="quantized_update",
                compressor=str(self.config.get("federated", {}).get("quantization_dtype", "float16")) + "_quantized_dense",
                privacy_clip_norm=privacy_clip_norm,
                privacy_noise_multiplier=privacy_noise_multiplier,
            )
        if compressed:
            update = subtract_state(local_state, global_state)
            fraction = float(self.config["federated"].get("topk_fraction", 0.05))
            payload_kind = "sparse_update"
            compressor = "topk"
            privacy_clip_norm = 0.0
            privacy_noise_multiplier = 0.0
            if algorithm in {"soteriafl", "dp_topk_fedavg"}:
                privacy_cfg = self.config.get("privacy", {})
                privacy_clip_norm = float(privacy_cfg.get("clip_norm", 1.0))
                privacy_noise_multiplier = float(privacy_cfg.get("noise_multiplier", 0.1))
                update = privatize_state_update(update, privacy_clip_norm, privacy_noise_multiplier)
                if algorithm == "soteriafl":
                    sparse = compress_randomk(update, fraction)
                    payload_kind = "soteriafl_randomk_dp_update"
                    compressor = "randomk_unbiased"
                else:
                    sparse = compress_topk(update, fraction)
                    payload_kind = "dp_topk_dp_update"
                    compressor = "topk_dp"
            else:
                sparse = compress_topk(update, fraction)
            return ClientResult(
                **common,
                sparse_update=sparse,
                upload_bytes=sparse.nbytes,
                upload_parameters=sparse.values.numel(),
                payload_kind=payload_kind,
                compressor=compressor,
                privacy_clip_norm=privacy_clip_norm,
                privacy_noise_multiplier=privacy_noise_multiplier,
            )
        dense_update = subtract_state(local_state, global_state)
        return ClientResult(
            **common,
            state=dense_update,
            upload_bytes=dense_bytes,
            upload_parameters=dense_parameters,
            payload_kind="dense_update",
        )

    def gradient_sample(self, global_state: StateDict, max_samples: int | None = None, batch_index: int = 0):
        """Return gradients for a selected batch for DLG/iDLG evaluation."""

        model = build_model(self.config).to(self.device)
        load_serialized(model, copy.deepcopy(global_state), self.device)
        return first_batch_gradient(
            model,
            self.train_loader,
            self.device,
            max_samples=max_samples,
            model_mode=str(self.config.get("attack", {}).get("model_mode", "train")),
            batch_index=batch_index,
        )

    def sample_batch(self, max_samples: int | None = None, batch_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one selected local batch for payload-based reconstruction attacks.

        Example:
            ``x, y = client.sample_batch(max_samples=1, batch_index=2)`` selects
            the third local mini-batch used for attack-side evaluation.
        """

        return first_batch_sample(self.train_loader, self.device, max_samples=max_samples, batch_index=batch_index)
