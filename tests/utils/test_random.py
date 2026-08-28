from __future__ import annotations

import fedlab.federated.algorithms as algorithms_module
from fedlab.utils.random import seed_cuda_device


def test_seed_cuda_device_targets_only_requested_device(monkeypatch):
    calls: list[tuple[str, object]] = []

    class _CudaDeviceContext:
        def __init__(self, device):
            self.device = str(device)

        def __enter__(self):
            calls.append(("device", self.device))
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(algorithms_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(algorithms_module.torch.cuda, "device", lambda device: _CudaDeviceContext(device))
    monkeypatch.setattr(algorithms_module.torch.cuda, "manual_seed", lambda seed: calls.append(("manual_seed", int(seed))))
    monkeypatch.setattr(algorithms_module.torch.cuda, "manual_seed_all", lambda seed: calls.append(("manual_seed_all", int(seed))), raising=False)

    seed_cuda_device(2026, "cuda:1")

    assert calls == [("device", "cuda:1"), ("manual_seed", 2026)]


def test_configure_random_seed_uses_runtime_device(monkeypatch):
    captured: dict[str, object] = {}

    def fake_setup_seed(seed, deterministic=True, *, device=None):
        captured["seed"] = int(seed)
        captured["deterministic"] = bool(deterministic)
        captured["device"] = str(device)

    monkeypatch.setattr(algorithms_module, "setup_seed", fake_setup_seed)
    monkeypatch.setattr(algorithms_module.torch.cuda, "is_available", lambda: True)

    algorithms_module.configure_random_seed({
        "runtime": {"seed": 7, "deterministic": True, "device": "cuda:1"},
    })

    assert captured == {"seed": 7, "deterministic": True, "device": "cuda:1"}
