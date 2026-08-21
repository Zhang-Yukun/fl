"""Helpers for estimating serialized transport overhead.

Example:
    ``estimate_download_transport_bytes(state, round_index=3, compressed=False)``
    measures the single-node serialized envelope size for one server download.
"""

from __future__ import annotations

import pickle
from typing import Any


def serialized_num_bytes(payload: Any) -> int:
    """Return the pickle-serialized size of one Python payload."""

    return len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def estimate_download_transport_bytes(
    state: Any,
    *,
    round_index: int,
    compressed: bool,
    round_context: dict[str, Any] | None = None,
    stop: bool = False,
    base_bytes: int = 0,
) -> int:
    """Estimate one download envelope size for local federated simulation."""

    payload = {
        "round": round_index,
        "state": state,
        "compressed": compressed,
        "round_context": round_context or {},
        "stop": stop,
    }
    return int(base_bytes) + serialized_num_bytes(payload)


def estimate_upload_transport_bytes(
    result: Any,
    *,
    round_index: int,
    base_bytes: int = 0,
    max_iters: int = 8,
) -> int:
    """Estimate one upload envelope size for local or gRPC client submission.

    The serialized payload includes the result object itself, so the
    ``transport_upload_*`` fields can affect the final byte count. A short fixed
    point iteration keeps the estimate self-consistent.
    """

    base_bytes = int(base_bytes)
    estimated = max(base_bytes + int(getattr(result, "parameter_upload_bytes", 0)), int(getattr(result, "transport_upload_bytes", 0)))
    for _ in range(max_iters):
        result.transport_upload_bytes = estimated
        result.transport_upload_overhead_bytes = max(0, estimated - int(getattr(result, "parameter_upload_bytes", 0)))
        candidate = base_bytes + serialized_num_bytes({"round": round_index, "result": result})
        if candidate == estimated:
            break
        estimated = candidate
    result.transport_upload_bytes = estimated
    result.transport_upload_overhead_bytes = max(0, estimated - int(getattr(result, "parameter_upload_bytes", 0)))
    return estimated
