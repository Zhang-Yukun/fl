"""Federated client implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from federated_ts.models import build_model
from federated_ts.serialization import SparseUpdate, StateDict, compress_topk, load_serialized, serialize_model, subtract_state
from federated_ts.training import first_batch_gradient, train_one_epoch


@dataclass
class ClientResult:
    client_id: str
    num_samples: int
    loss: float
    state: StateDict | None = None
    sparse_update: SparseUpdate | None = None
    dense_bytes: int = 0
    sent_bytes: int = 0


class FederatedClient:
    """Local trainer that receives global parameters and returns an update."""

    def __init__(self, client_id: str, train_loader, config: dict[str, Any], device: torch.device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.config = config
        self.device = device

    def train(self, global_state: StateDict, compressed: bool = False) -> ClientResult:
        model = build_model(self.config).to(self.device)
        load_serialized(model, global_state, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config["training"].get("lr", 1e-3)))
        losses = []
        for _ in range(int(self.config["federated"].get("local_epochs", 1))):
            losses.append(train_one_epoch(model, self.train_loader, optimizer, self.device))
        local_state = serialize_model(model)
        dense_bytes = sum(t.numel() * t.element_size() for t in local_state.values())
        if compressed:
            update = subtract_state(local_state, global_state)
            sparse = compress_topk(update, float(self.config["federated"].get("topk_fraction", 0.05)))
            return ClientResult(self.client_id, len(self.train_loader.dataset), float(sum(losses) / len(losses)), sparse_update=sparse, dense_bytes=dense_bytes, sent_bytes=sparse.nbytes)
        return ClientResult(self.client_id, len(self.train_loader.dataset), float(sum(losses) / len(losses)), state=local_state, dense_bytes=dense_bytes, sent_bytes=dense_bytes)

    def gradient_sample(self, global_state: StateDict):
        model = build_model(self.config).to(self.device)
        load_serialized(model, copy.deepcopy(global_state), self.device)
        return first_batch_gradient(model, self.train_loader, self.device)

