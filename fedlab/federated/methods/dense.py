"""Dense federated algorithm implementations backed by the new method API."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from fedlab.federated.methods.base import FederatedMethod, MethodCapabilities
from fedlab.federated.methods.registry import federated_method
from fedlab.federated.protocol import build_upload_payload_state, derive_update_from_upload_payload, resolve_upload_mode, weighted_protocol_base_state
from fedlab.utils.aggregation import fedaware_weights
from fedlab.utils.privacy_accounting import AdaptiveClippedRdpAccountant
from fedlab.utils.serialization import add_update, average_states, subtract_state, state_num_bytes, state_num_parameters


class _DenseFedAvgMethod(FederatedMethod):
    """Shared dense-update behavior for FedAvg-style algorithms."""

    def client_update(
        self,
        *,
        local_state,
        global_state,
        received_global_state,
        common: dict[str, Any],
        evaluation_kwargs: dict[str, Any],
        result_cls,
        client=None,
        **_: Any,
    ):
        """Return a dense semantic payload from one local model state."""

        upload_mode = resolve_upload_mode(client.config if client is not None else {})
        base_state = received_global_state if received_global_state is not None else global_state
        payload_state = build_upload_payload_state(local_state, base_state, upload_mode)
        payload_bytes = state_num_bytes(payload_state)
        payload_parameters = state_num_parameters(payload_state)
        return result_cls(
            **common,
            aggregation_state=payload_state,
            **evaluation_kwargs,
            upload_mode=upload_mode,
            upload_bytes=payload_bytes,
            upload_parameters=payload_parameters,
            parameter_upload_bytes=payload_bytes,
            parameter_upload_parameters=payload_parameters,
            transport_upload_bytes=payload_bytes,
            aggregation_payload_kind='dense_model' if upload_mode == 'model' else 'dense_update',
        )

    def extract_attack_payload(self, *, result, clone_state, server=None, round_base_state=None, round_index: int = 0, round_context=None, **_: Any):
        """Expose the dense uploaded client payload as a derived update to the server attacker."""

        if result.aggregation_state is None:
            raise ValueError(f'Client {result.client_id} did not produce an attackable dense payload')
        payload_state = clone_state(result.aggregation_state)
        if result.upload_mode == 'model':
            if server is None or round_base_state is None:
                raise ValueError('Model-mode dense attack extraction requires server context and round base state')
            base_state = server.method.reconstruct_received_global_state(
                server=server,
                global_state=round_base_state,
                client_id=result.client_id,
                round_index=round_index,
                round_context=round_context or {},
            )
            return derive_update_from_upload_payload(payload_state, base_state, result.upload_mode)
        return payload_state


@federated_method('fedavg', compressed=False, description='Standard dense FedAvg baseline')
class FedAvgMethod(_DenseFedAvgMethod):
    """Concrete dense FedAvg implementation on the method API."""

    name = 'fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Standard dense FedAvg baseline')

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: Any) -> list[float]:
        """Aggregate dense client payloads with sample-size-weighted FedAvg."""

        sample_weights = [result.num_samples for result in results]
        weights = [weight / float(sum(sample_weights)) for weight in sample_weights]
        updates = [
            derive_update_from_upload_payload(
                result.aggregation_state,
                server.method.reconstruct_received_global_state(
                    server=server,
                    global_state=round_base_state,
                    client_id=result.client_id,
                    round_index=round_index,
                    round_context=round_context or {},
                ),
                result.upload_mode,
            )
            for result in results
        ]
        averaged_update = average_states(updates, sample_weights)
        protocol_base_state = weighted_protocol_base_state(server, results, round_base_state, round_index, round_context or {})
        server.global_state = add_update(protocol_base_state, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        return weights


@federated_method('fedaware', compressed=False, description='Adaptive weighted dense FedAvg aggregation')
class FedAwareMethod(_DenseFedAvgMethod):
    """Concrete FedAware implementation on the method API."""

    name = 'fedaware'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Adaptive weighted dense FedAvg aggregation')

    def aggregate(self, *, server, results, round_base_state=None, round_index: int = 0, round_context=None, **_: Any) -> list[float]:
        """Aggregate dense updates with FedAware's learned aggregation weights."""

        aware_cfg = server.config.get('fedaware', {})
        sample_weights = [result.num_samples for result in results]
        updates = [
            derive_update_from_upload_payload(
                result.aggregation_state,
                server.method.reconstruct_received_global_state(
                    server=server,
                    global_state=round_base_state,
                    client_id=result.client_id,
                    round_index=round_index,
                    round_context=round_context or {},
                ),
                result.upload_mode,
            )
            for result in results
        ]
        weights = fedaware_weights(
            updates,
            sample_weights,
            alpha=float(aware_cfg.get('alpha', 0.5)),
            steps=int(aware_cfg.get('steps', 50)),
            lr=float(aware_cfg.get('lr', 0.1)),
        )
        averaged_update = None
        for update, weight in zip(updates, weights):
            scaled = OrderedDict((name, tensor * weight) for name, tensor in update.items())
            averaged_update = scaled if averaged_update is None else OrderedDict(
                (name, averaged_update[name] + scaled[name]) for name in scaled
            )
        protocol_base_state = weighted_protocol_base_state(server, results, round_base_state, round_index, round_context or {})
        server.global_state = add_update(protocol_base_state, averaged_update)
        server._update_oracle_evaluation_state(round_base_state, results, sample_weights)
        return weights


@federated_method('adaptive_clipped_rdp_fedavg', compressed=False, description='Dense FedAvg with adaptive clipping and RDP accounting')
class AdaptiveClippedRdpFedAvgMethod(_DenseFedAvgMethod):
    """Concrete adaptive clipped RDP FedAvg implementation on the method API."""

    name = 'adaptive_clipped_rdp_fedavg'
    capabilities = MethodCapabilities(compressed=False, implemented=True, description='Dense FedAvg with adaptive clipping and RDP accounting')

    def configure_server(self, server: Any) -> None:
        """Initialize the adaptive RDP accountant on the server."""

        adaptive_cfg = server.config.get('adaptive_clipped_rdp', {})
        server.adaptive_accountant = AdaptiveClippedRdpAccountant(
            rdp_alpha=float(adaptive_cfg.get('rdp_alpha', 16.0)),
            delta=float(adaptive_cfg.get('delta', 1e-5)),
            noise_multiplier=float(adaptive_cfg.get('noise_multiplier', 0.0)),
        )

    def aggregate(self, *, server, results, round_index=0, round_base_state=None, round_context=None, **_: Any) -> list[float]:
        """Delegate to the existing adaptive clipped aggregation routine."""

        return server._aggregate_adaptive_clipped_rdp(results, round_index, round_base_state, round_context or {})
