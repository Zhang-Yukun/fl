"""Federated client implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from federated_ts.models import build_model
from federated_ts.serialization import (
    SparseUpdate,
    StateDict,
    compress_topk,
    load_serialized,
    serialize_model,
    state_num_bytes,
    state_num_parameters,
    subtract_state,
)
from federated_ts.training import first_batch_gradient, train_one_epoch


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
        """Train locally from global weights and return a dense or sparse payload.

        Example:
            ``client.train(state, compressed=False)`` implements standard FedAvg
            client behavior by uploading a full dense model state.
        """

        model = build_model(self.config).to(self.device)
        load_serialized(model, global_state, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config["training"].get("lr", 1e-3)))
        losses = []
        for _ in range(int(self.config["federated"].get("local_epochs", 1))):
            losses.append(train_one_epoch(model, self.train_loader, optimizer, self.device))
        local_state = serialize_model(model)
        dense_bytes = state_num_bytes(local_state)
        dense_parameters = state_num_parameters(local_state)
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
            sparse = compress_topk(update, float(self.config["federated"].get("topk_fraction", 0.05)))
            return ClientResult(
                **common,
                sparse_update=sparse,
                upload_bytes=sparse.nbytes,
                upload_parameters=sparse.values.numel(),
                payload_kind="sparse_update",
            )
        return ClientResult(
            **common,
            state=local_state,
            upload_bytes=dense_bytes,
            upload_parameters=dense_parameters,
            payload_kind="dense_state",
        )

    def gradient_sample(self, global_state: StateDict):
        """Return first-batch gradients for DLG/iDLG evaluation.

        Example:
            ``grads, x, y = client.gradient_sample(server_state)``.
        """

        model = build_model(self.config).to(self.device)
        load_serialized(model, copy.deepcopy(global_state), self.device)
        return first_batch_gradient(model, self.train_loader, self.device)
