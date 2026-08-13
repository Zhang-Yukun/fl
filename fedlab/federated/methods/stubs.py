"""Current federated algorithm name registry as migration stubs.

These classes keep the registry complete for algorithms that have not yet been
migrated into standalone method modules.
"""

from __future__ import annotations

from fedlab.federated.methods.base import FederatedMethod, MethodCapabilities
from fedlab.federated.methods.registry import federated_method


class _LegacyStubMethod(FederatedMethod):
    """Temporary placeholder while the runtime is migrated to the method API."""

    def client_update(self, **kwargs):  # pragma: no cover - guard during migration
        """Raise until this method is migrated to the pluggable runtime."""

        raise RuntimeError(f"{self.name} has not been migrated to the pluggable runtime yet")

    def aggregate(self, **kwargs):  # pragma: no cover - guard during migration
        """Raise until this method is migrated to the pluggable runtime."""

        raise RuntimeError(f"{self.name} has not been migrated to the pluggable runtime yet")

    def extract_attack_payload(self, **kwargs):  # pragma: no cover - guard during migration
        """Raise until this method is migrated to the pluggable runtime."""

        raise RuntimeError(f"{self.name} has not been migrated to the pluggable runtime yet")


@federated_method('ega_fedavg', compressed=False, description='Encoded gradient aggregation FedAvg variant')
class EGAFedAvgStub(_LegacyStubMethod):
    """Registered EGA FedAvg stub for the migration registry."""

    name = 'ega_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=False, description='Encoded gradient aggregation FedAvg variant')
