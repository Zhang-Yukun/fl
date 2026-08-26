"""Shared helpers for quantized dense federated methods."""

from __future__ import annotations

from typing import Any

from fedlab.federated.methods.base import FederatedMethod


class QuantizedDenseMethodBase(FederatedMethod):
    """Shared helpers for dense-upload quantized FedAvg variants."""

    def _dense_attack_payload(self, *, result, **_: Any):
        """Return the uploaded dense payload view exposed to the server."""

        if result.aggregation_state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        return result.aggregation_state
