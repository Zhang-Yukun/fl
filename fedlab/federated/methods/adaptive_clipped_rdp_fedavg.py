"""Adaptive clipped RDP FedAvg implementation."""

from __future__ import annotations

from typing import Any

from fedlab.federated.methods._dense_common import DenseFedAvgMethodBase
from fedlab.federated.methods.base import MethodCapabilities, MethodConfigSpec
from fedlab.federated.methods.registry import federated_method
from fedlab.utils.privacy_accounting import AdaptiveClippedRdpAccountant


@federated_method('adaptive_clipped_rdp_fedavg', compressed=False, description='Dense FedAvg with adaptive clipping and RDP accounting')
class AdaptiveClippedRdpFedAvgMethod(DenseFedAvgMethodBase):
    """Concrete adaptive clipped RDP FedAvg implementation on the method API."""

    name = 'adaptive_clipped_rdp_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Dense FedAvg with adaptive clipping and RDP accounting')
    config_spec = MethodConfigSpec(root_blocks=frozenset({'adaptive_clipped_rdp'}))

    def configure_server(self, server: Any) -> None:
        """Initialize the adaptive RDP accountant on the server."""

        adaptive_cfg = server.config.get('adaptive_clipped_rdp', {})
        server.adaptive_accountant = AdaptiveClippedRdpAccountant(
            rdp_alpha=float(adaptive_cfg.get('rdp_alpha', 16.0)),
            delta=float(adaptive_cfg.get('delta', 1e-5)),
            noise_multiplier=float(adaptive_cfg.get('noise_multiplier', 0.0)),
        )

    def aggregate(self, *, server, results, round_index=0, round_base_state=None, round_context=None, **_: object) -> list[float]:
        """Delegate to the existing adaptive clipped aggregation routine."""

        return server._aggregate_adaptive_clipped_rdp(results, round_index, round_base_state, round_context or {})
