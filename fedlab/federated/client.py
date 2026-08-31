"""Federated client implementation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from fedlab.modeling.ega import EncodedStatePayload, encode_state_update
from fedlab.federated.methods import build_method
from fedlab.federated.protocol import build_download_payload_state, reconstruct_model_from_download_payload
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
from fedlab.utils.transport import (
    auxiliary_payload_num_bytes,
    auxiliary_payload_num_parameters,
    estimate_download_transport_bytes,
    estimate_upload_transport_bytes,
)
from fedlab.engine.training import build_training_optimizer, first_batch_sample, train_n_steps, train_one_epoch


@dataclass
class PreparedClientState:
    """Per-round download payload and reconstructed client-visible model."""

    download_state: StateDict
    received_global_state: StateDict


@dataclass
class ClientResult:
    """Payload and communication metadata returned by one client update."""

    client_id: str
    num_samples: int
    loss: float
    aggregation_state: StateDict | None = None
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
    aggregation_payload_kind: str = "dense_update"
    compressor: str = "none"
    privacy_clip_norm: float = 0.0
    privacy_noise_multiplier: float = 0.0

    @property
    def sent_bytes(self) -> int:
        """Backward-compatible alias for upload bytes."""

        return self.upload_bytes

    @property
    def state(self) -> StateDict | None:
        """Backward-compatible alias for the aggregation-visible dense payload."""

        return self.aggregation_state

    @state.setter
    def state(self, value: StateDict | None) -> None:
        """Keep legacy callers functional while the codebase uses explicit names."""

        self.aggregation_state = value

    @property
    def payload_kind(self) -> str:
        """Backward-compatible alias for the aggregation payload semantic label."""

        return self.aggregation_payload_kind

    @payload_kind.setter
    def payload_kind(self, value: str) -> None:
        """Keep legacy callers functional while the codebase uses explicit names."""

        self.aggregation_payload_kind = value



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
        self.cached_received_global_state = None
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


    def apply_global_training_context(self, round_context: dict[str, Any] | None = None) -> None:
        """Update cached global training statistics from one round context payload."""

        if not round_context:
            return
        total_train_samples = round_context.get('total_train_samples')
        total_clients = round_context.get('total_clients')
        if total_train_samples is not None:
            self.total_train_samples = int(total_train_samples)
        if total_clients is not None:
            self.total_clients = int(total_clients)


    def prepare_round_state(
        self,
        global_state: StateDict,
        round_index: int = 0,
        round_context: dict[str, Any] | None = None,
    ) -> PreparedClientState:
        """Return the download payload and client-visible model for one round."""

        method_download_state, target_received_global_state = self.method.prepare_client_state(
            client=self,
            global_state=global_state,
            round_index=round_index,
            round_context=round_context or {},
        )
        if self.method.uses_custom_download_transport():
            download_state = method_download_state
            received_global_state = target_received_global_state
        else:
            download_base_state = self.cached_received_global_state if self.cached_received_global_state is not None else global_state
            download_state = build_download_payload_state(target_received_global_state, download_base_state)
            received_global_state = reconstruct_model_from_download_payload(download_state, download_base_state)
        self.cached_received_global_state = received_global_state
        return PreparedClientState(download_state=download_state, received_global_state=received_global_state)

    def train(
        self,
        global_state: StateDict,
        compressed: bool = False,
        round_index: int = 0,
        round_context: dict[str, Any] | None = None,
        prepared_state: PreparedClientState | None = None,
    ) -> ClientResult:
        """Train locally from global weights and return the transmitted payload.

        Example:
            Standard FedAvg-style clients upload dense model updates
            ``local_state - received_global_state`` instead of full model states.
        """

        algorithm = str(self.config["federated"].get("algorithm", "fedavg")).lower()
        self.apply_global_training_context(round_context)
        if prepared_state is None:
            prepared_state = self.prepare_round_state(global_state, round_index=round_index, round_context=round_context)
        download_state = prepared_state.download_state
        received_global_state = prepared_state.received_global_state

        model = build_model(self.config).to(self.device)
        load_serialized(model, received_global_state, self.device)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = build_training_optimizer(trainable_parameters, self.config)
        losses = []
        local_steps = self.config["federated"].get("local_steps")
        if local_steps is not None:
            losses.append(train_n_steps(model, self.train_loader, optimizer, self.device, int(local_steps)))
        else:
            for _ in range(int(self.config.get("training", {}).get("epochs", 1))):
                losses.append(train_one_epoch(model, self.train_loader, optimizer, self.device))
        local_state = serialize_model(model)
        dense_bytes = state_num_bytes(local_state)
        dense_parameters = state_num_parameters(local_state)
        round_context_payload = round_context or {}
        algorithm_download_bytes = state_num_bytes(download_state) + auxiliary_payload_num_bytes(round_context_payload)
        algorithm_download_parameters = state_num_parameters(download_state) + auxiliary_payload_num_parameters(round_context_payload)
        common = dict(
            client_id=self.client_id,
            num_samples=self._loader_num_samples(self.train_loader),
            loss=float(sum(losses) / len(losses)),
            dense_bytes=dense_bytes,
            dense_parameters=dense_parameters,
            download_bytes=algorithm_download_bytes,
            download_parameters=algorithm_download_parameters,
            parameter_download_bytes=algorithm_download_bytes,
            parameter_download_parameters=algorithm_download_parameters,
            transport_download_bytes=estimate_download_transport_bytes(
                download_state,
                round_index=round_index,
                compressed=compressed,
                round_context=round_context_payload,
            ),
            transport_download_overhead_bytes=0,
            dense_download_reference_bytes=state_num_bytes(global_state),
            dense_download_reference_parameters=state_num_parameters(global_state),
        )
        common["transport_download_overhead_bytes"] = max(0, common["transport_download_bytes"] - common["parameter_download_bytes"])
        result = self.method.client_update(
            client=self,
            model=model,
            local_state=local_state,
            global_state=global_state,
            received_global_state=received_global_state,
            download_state=download_state,
            round_index=round_index,
            round_context=round_context or {},
            common=common,
            evaluation_kwargs={},
            result_cls=ClientResult,
        )
        estimate_upload_transport_bytes(result, round_index=round_index)
        return result


    def sample_batch(self, max_samples: int | None = None, batch_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one selected local batch for payload-based reconstruction attacks.

        Example:
            ``x, y = client.sample_batch(max_samples=1, batch_index=2)`` selects
            the third local mini-batch used for attack-side evaluation.
        """

        return first_batch_sample(self.train_loader, self.device, max_samples=max_samples, batch_index=batch_index)

    def train_num_samples(self) -> int:
        """Return the number of local training samples."""

        dataset = self.train_loader.dataset
        return int(len(dataset))

    def train_reference_inputs(self) -> torch.Tensor:
        """Return all normalized input windows from this client training dataset.

        Example:
            ``windows = client.train_reference_inputs()`` is used to score
            payload reconstructions against the full local training set.
        """

        dataset = self.train_loader.dataset
        return torch.stack([dataset[index][0] for index in range(len(dataset))], dim=0)

    def train_reference_targets(self) -> torch.Tensor:
        """Return all normalized target windows from this client training dataset.

        Example:
            ``targets = client.train_reference_targets()`` provides the paired
            target windows for reference-aligned attack visualization.
        """

        dataset = self.train_loader.dataset
        return torch.stack([dataset[index][1] for index in range(len(dataset))], dim=0)
