"""Base interfaces for pluggable federated algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class MethodCapabilities:
    """Static capability flags exposed by one federated algorithm.

    Example:
        ``MethodCapabilities(compressed=True, implemented=True, description="Top-k sparse upload")``
        marks one method as a migrated compressed communication variant.
    """

    compressed: bool = False
    implemented: bool = False
    description: str = ""


@dataclass(frozen=True)
class MethodConfigSpec:
    """Config metadata exposed by one federated algorithm plugin."""

    federated_keys: frozenset[str] = frozenset()
    root_blocks: frozenset[str] = frozenset()
    uses_privacy_block: bool = False


class FederatedMethod(ABC):
    """Abstract interface implemented by all federated algorithm modules.

    The runtime should only depend on this interface and the registry instead of
    dispatching on algorithm-name string branches.
    """

    name: ClassVar[str]
    capabilities: ClassVar[MethodCapabilities] = MethodCapabilities()
    config_spec: ClassVar[MethodConfigSpec] = MethodConfigSpec()

    def configure_client(self, client: Any) -> None:
        """Attach algorithm-specific state to a client before training starts."""

    def configure_server(self, server: Any) -> None:
        """Attach algorithm-specific state to a server before training starts."""

    def build_round_context(self, server: Any) -> dict[str, Any]:
        """Return any per-round context the server must broadcast to clients."""

        return {}

    def sync_server_client_state(self, *, server: Any, clients: list[Any]) -> None:
        """Copy any client-side per-round state the server needs for exact protocol reconstruction."""

    def uses_custom_download_transport(self) -> bool:
        """Return whether the method owns download-payload construction itself."""

        return False

    def prepare_client_state(
        self,
        *,
        client: Any,
        global_state: Any,
        round_index: int,
        round_context: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Return the transmitted download state and the state loaded for local training."""

        return global_state, global_state

    def reconstruct_received_global_state(
        self,
        *,
        server: Any,
        global_state: Any,
        client_id: str,
        round_index: int,
        round_context: dict[str, Any],
    ) -> Any:
        """Return the model state a client effectively trains from this round."""

        return global_state

    @abstractmethod
    def client_update(self, **kwargs: Any) -> Any:
        """Build the algorithm-specific client payload from one local update."""

    @abstractmethod
    def aggregate(self, **kwargs: Any) -> list[float]:
        """Aggregate one round of client payloads and return client weights."""

    @abstractmethod
    def extract_attack_payload(self, **kwargs: Any) -> Any:
        """Return the true payload view exposed to the server-side attacker."""
