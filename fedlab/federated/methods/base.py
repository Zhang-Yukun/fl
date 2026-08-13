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


class FederatedMethod(ABC):
    """Abstract interface implemented by all federated algorithm modules.

    The runtime should only depend on this interface and the registry instead of
    dispatching on algorithm-name string branches.
    """

    name: ClassVar[str]
    capabilities: ClassVar[MethodCapabilities] = MethodCapabilities()

    def configure_client(self, client: Any) -> None:
        """Attach algorithm-specific state to a client before training starts."""

    def configure_server(self, server: Any) -> None:
        """Attach algorithm-specific state to a server before training starts."""

    def build_round_context(self, server: Any) -> dict[str, Any]:
        """Return any per-round context the server must broadcast to clients."""

        return {}

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

    @abstractmethod
    def client_update(self, **kwargs: Any) -> Any:
        """Build the algorithm-specific client payload from one local update."""

    @abstractmethod
    def aggregate(self, **kwargs: Any) -> list[float]:
        """Aggregate one round of client payloads and return client weights."""

    @abstractmethod
    def extract_attack_payload(self, **kwargs: Any) -> Any:
        """Return the true payload view exposed to the server-side attacker."""
