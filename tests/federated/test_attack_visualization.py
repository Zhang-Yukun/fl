from types import SimpleNamespace

import torch

from fedlab.federated.algorithms import _log_attack_reconstruction_views


class _TrackerStub:
    def __init__(self):
        self.attack_images = []

    def log_attack_reconstruction(self, key, result, step=None):
        self.attack_images.append((key, step, result))


def _attack_result_stub(*, name: str, client_id: str, reconstructed_x: torch.Tensor, reference_x: torch.Tensor, matched_indices: list[int]):
    zeros_y = torch.zeros(reconstructed_x.shape[0], 1, 1)
    return SimpleNamespace(
        name=name,
        client_id=client_id,
        round_index=0,
        sample_index=0,
        reconstructed_x=reconstructed_x,
        reference_x=reference_x,
        reconstructed_y=zeros_y.clone(),
        reference_y=zeros_y.clone(),
        plot_reconstructed_x=reconstructed_x.clone(),
        plot_reference_x=reference_x.clone(),
        plot_reconstructed_y=zeros_y.clone(),
        plot_reference_y=zeros_y.clone(),
        matched_reference_indices=matched_indices,
        matched_reference_metric_name="mse",
    )


def test_log_attack_reconstruction_views_logs_only_best_matched_sample():
    tracker = _TrackerStub()
    config = {"attack": {"data_range": 1.0}}
    worse = _attack_result_stub(
        name="DLG",
        client_id="client1",
        reconstructed_x=torch.tensor([[[5.0], [5.0]], [[1.5], [2.5]]]),
        reference_x=torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]]]),
        matched_indices=[8, 9],
    )
    best = _attack_result_stub(
        name="DLG",
        client_id="client1",
        reconstructed_x=torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        reference_x=torch.tensor([[[1.0], [2.0]], [[0.0], [0.0]]]),
        matched_indices=[3, 4],
    )

    _log_attack_reconstruction_views(tracker, [worse, best], step=10, config=config)

    assert [entry[0] for entry in tracker.attack_images] == [
        "attack/DLG/reconstruction",
        "attack/client/client1/DLG/reconstruction",
    ]
    for key, step, result in tracker.attack_images:
        assert step == 10
        assert result.reconstructed_x.shape == (1, 2, 1), key
        assert torch.equal(result.reconstructed_x, torch.tensor([[[1.0], [2.0]]])), key
        assert torch.equal(result.reference_x, torch.tensor([[[1.0], [2.0]]])), key
        assert result.matched_reference_indices == [3], key
        assert result.matched_reference_metric_value == 0.0, key
        assert result.matched_reference_metric_min_value == 0.0, key
