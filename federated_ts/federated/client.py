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
    load_serialized,
    privatize_state_update,
    serialize_model,
    state_num_bytes,
    state_num_parameters,
    subtract_state,
)
from federated_ts.engine.training import first_batch_gradient, train_one_epoch


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
    upload_bytes: int = 0
    upload_parameters: int = 0
    payload_kind: str = "dense_state"
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

    def train(self, global_state: StateDict, compressed: bool = False) -> ClientResult:
        """Train locally from global weights and return a dense or sparse payload."""

        model = build_model(self.config).to(self.device)
        load_serialized(model, global_state, self.device)
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
            download_bytes=state_num_bytes(global_state),
            download_parameters=state_num_parameters(global_state),
        )
        if compressed:
            update = subtract_state(local_state, global_state)
            fraction = float(self.config["federated"].get("topk_fraction", 0.05))
            payload_kind = "sparse_update"
            compressor = "topk"
            privacy_clip_norm = 0.0
            privacy_noise_multiplier = 0.0
            if str(self.config["federated"].get("algorithm", "fedavg")).lower() == "soteriafl":
                privacy_cfg = self.config.get("privacy", {})
                privacy_clip_norm = float(privacy_cfg.get("clip_norm", 1.0))
                privacy_noise_multiplier = float(privacy_cfg.get("noise_multiplier", 0.1))
                update = privatize_state_update(update, privacy_clip_norm, privacy_noise_multiplier)
                sparse = compress_randomk(update, fraction)
                payload_kind = "soteriafl_randomk_dp_update"
                compressor = "randomk_unbiased"
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
        return ClientResult(
            **common,
            state=local_state,
            upload_bytes=dense_bytes,
            upload_parameters=dense_parameters,
            payload_kind="dense_state",
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
