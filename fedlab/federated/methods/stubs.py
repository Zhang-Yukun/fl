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


@federated_method('compressed_fedavg', compressed=True, description='Legacy top-k sparse FedAvg alias')
class CompressedFedAvgStub(_LegacyStubMethod):
    """Registered legacy compressed FedAvg stub for the migration registry."""

    name = 'compressed_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=False, description='Legacy top-k sparse FedAvg alias')


@federated_method('sparse_fedavg', compressed=True, description='Top-k sparse FedAvg baseline')
class SparseFedAvgStub(_LegacyStubMethod):
    """Registered sparse FedAvg stub for the migration registry."""

    name = 'sparse_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=False, description='Top-k sparse FedAvg baseline')


@federated_method('dp_topk_fedavg', compressed=True, description='Top-k sparse FedAvg with DP preprocessing')
class DpTopkFedAvgStub(_LegacyStubMethod):
    """Registered DP Top-k FedAvg stub for the migration registry."""

    name = 'dp_topk_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=False, description='Top-k sparse FedAvg with DP preprocessing')


@federated_method('randomk_fedavg', compressed=True, description='Random-k sparse FedAvg baseline')
class RandomkFedAvgStub(_LegacyStubMethod):
    """Registered Random-k FedAvg stub for the migration registry."""

    name = 'randomk_fedavg'
    capabilities = MethodCapabilities(compressed=True, implemented=False, description='Random-k sparse FedAvg baseline')


@federated_method('soteriafl', compressed=True, description='Private random-k sparse upload baseline')
class SoteriaFLStub(_LegacyStubMethod):
    """Registered SoteriaFL stub for the migration registry."""

    name = 'soteriafl'
    capabilities = MethodCapabilities(compressed=True, implemented=False, description='Private random-k sparse upload baseline')


@federated_method('ega_fedavg', compressed=False, description='Encoded gradient aggregation FedAvg variant')
class EGAFedAvgStub(_LegacyStubMethod):
    """Registered EGA FedAvg stub for the migration registry."""

    name = 'ega_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=False, description='Encoded gradient aggregation FedAvg variant')
