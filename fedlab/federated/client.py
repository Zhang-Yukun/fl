"""Federated client implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from fedlab.modeling.ega import EncodedStatePayload, encode_state_update
from fedlab.federated.methods import build_method
from fedlab.modeling import build_model
from fedlab.utils.serialization import (
    SparseUpdate,
    StateDict,
    load_serialized,
    serialize_model,
    state_num_bytes,
    state_num_parameters,
    subtract_state,
)
from fedlab.engine.training import build_training_optimizer, first_batch_gradient, first_batch_sample, train_n_steps, train_one_epoch


@dataclass
class ClientResult:
    """Payload and communication metadata returned by one client update."""

    client_id: str
    num_samples: int
    loss: float
    state: StateDict | None = None
    sparse_update: SparseUpdate | None = None
    ega_payload: EncodedStatePayload | None = None
    dense_bytes: int = 0
    dense_parameters: int = 0
    download_bytes: int = 0
    download_parameters: int = 0
    parameter_download_bytes: int = 0
    parameter_download_parameters: int = 0
    dense_download_reference_bytes: int = 0
    dense_download_reference_parameters: int = 0
    upload_bytes: int = 0
    upload_parameters: int = 0
    parameter_upload_bytes: int = 0
    parameter_upload_parameters: int = 0
    transport_download_bytes: int = 0
    transport_upload_bytes: int = 0
    transport_download_overhead_bytes: int = 0
    transport_upload_overhead_bytes: int = 0
    payload_kind: str = "dense_update"
    compressor: str = "none"
    privacy_clip_norm: float = 0.0
    privacy_noise_multiplier: float = 0.0
    evaluation_update: StateDict | None = None
    evaluation_payload_kind: str = "none"

    @property
    def sent_bytes(self) -> int:
        """Backward-compatible alias for upload bytes."""

        return self.upload_bytes


class FederatedClient:
    """Local trainer that receives global parameters and returns an update."""

    def __init__(
        self,
        client_id: str,
        train_loader,
        config: dict[str, Any],
        device: torch.device,
        total_train_samples: int | None = None,
        total_clients: int | None = None,
        allow_ega_pretrain: bool = False,
    ):
        """Create a client bound to one local training loader."""

        self.client_id = client_id
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.total_train_samples = total_train_samples if total_train_samples is not None else self._loader_num_samples(train_loader)
        self.total_clients = total_clients if total_clients is not None else 1
        self.method = build_method(str(config.get("federated", {}).get("algorithm", "fedavg")))
        self.allow_ega_pretrain = allow_ega_pretrain
        self.ega_codec = None
        self.method.configure_client(self)

    @staticmethod
    def _loader_num_samples(loader) -> int:
        """Return the number of samples carried by one loader-like object."""

        dataset = getattr(loader, "dataset", None)
        return len(dataset) if dataset is not None else len(loader)

    def _upload_quantization_generator(self, round_index: int) -> torch.Generator | None:
        """Create a deterministic per-client generator for randomized upload quantization."""

        seed = self.config.get("federated", {}).get("quantization_seed")
        if seed is None:
            return None
        offset = sum(ord(char) for char in self.client_id)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + round_index * 1000 + offset)
        return generator

    def _randomk_generator(self, round_index: int) -> torch.Generator | None:
        """Create a deterministic per-client generator for random-k sparsification."""

        seed = self.config.get("federated", {}).get("randomk_seed")
        if seed is None:
            seed = self.config.get("runtime", {}).get("seed")
        if seed is None:
            return None
        offset = sum(ord(char) for char in self.client_id)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + round_index * 2000 + offset)
        return generator

    def _evaluation_result_kwargs(self, local_state: StateDict, global_state: StateDict) -> dict[str, Any]:
        """Return optional evaluation payloads used only by oracle evaluation mode."""

        evaluation_mode = str(self.config.get("evaluation", {}).get("mode", "protocol")).lower()
        if evaluation_mode != "oracle_full_update":
            return {}
        return {
            "evaluation_update": subtract_state(local_state, global_state),
            "evaluation_payload_kind": "dense_full_update",
        }

    def train(
        self,
        global_state: StateDict,
        compressed: bool = False,
        round_index: int = 0,
        round_context: dict[str, Any] | None = None,
    ) -> ClientResult:
        """Train locally from global weights and return the transmitted payload.

        Example:
            Standard FedAvg-style clients upload dense model updates
            ``local_state - received_global_state`` instead of full model states.
        """

        algorithm = str(self.config["federated"].get("algorithm", "fedavg")).lower()
        download_state, received_global_state = self.method.prepare_client_state(
            client=self,
            global_state=global_state,
            round_index=round_index,
            round_context=round_context or {},
        )

        model = build_model(self.config).to(self.device)
        load_serialized(model, received_global_state, self.device)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = build_training_optimizer(trainable_parameters, self.config)
        losses = []
        local_steps = self.config["federated"].get("local_steps")
        if local_steps is not None:
            losses.append(train_n_steps(model, self.train_loader, optimizer, self.device, int(local_steps)))
        else:
            for _ in range(int(self.config["federated"].get("local_epochs", 1))):
                losses.append(train_one_epoch(model, self.train_loader, optimizer, self.device))
        local_state = serialize_model(model)
        evaluation_kwargs = self._evaluation_result_kwargs(local_state, global_state)
        dense_bytes = state_num_bytes(local_state)
        dense_parameters = state_num_parameters(local_state)
        common = dict(
            client_id=self.client_id,
            num_samples=self._loader_num_samples(self.train_loader),
            loss=float(sum(losses) / len(losses)),
            dense_bytes=dense_bytes,
            dense_parameters=dense_parameters,
            download_bytes=state_num_bytes(download_state),
            download_parameters=state_num_parameters(download_state),
            parameter_download_bytes=state_num_bytes(download_state),
            parameter_download_parameters=state_num_parameters(download_state),
            transport_download_bytes=state_num_bytes(download_state),
            dense_download_reference_bytes=state_num_bytes(global_state),
            dense_download_reference_parameters=state_num_parameters(global_state),
        )
        return self.method.client_update(
            client=self,
            model=model,
            local_state=local_state,
            global_state=global_state,
            received_global_state=received_global_state,
            download_state=download_state,
            round_index=round_index,
            round_context=round_context or {},
            common=common,
            evaluation_kwargs=evaluation_kwargs,
            result_cls=ClientResult,
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

    def train_reference_inputs(self) -> torch.Tensor:
        """Return all normalized input windows from this client training dataset.

        Example:
            ``windows = client.train_reference_inputs()`` is used to score
            payload reconstructions against the full local training set.
        """

        dataset = self.train_loader.dataset
        return torch.stack([dataset[index][0] for index in range(len(dataset))], dim=0)
