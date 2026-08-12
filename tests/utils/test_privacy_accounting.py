import math
from collections import OrderedDict

import pytest
import torch

from fedlab.utils.privacy_accounting import (
    AdaptiveClippedRdpAccountant,
    adaptive_clip_threshold,
    clip_state_to_norm,
    state_l2_norm,
)


def test_adaptive_clip_threshold_uses_median_and_bounds():
    median_norm, raw_clip, clip_norm = adaptive_clip_threshold([1.0, 3.0, 5.0], 1.5, 0.5, 4.0)

    assert median_norm == pytest.approx(3.0)
    assert raw_clip == pytest.approx(4.5)
    assert clip_norm == pytest.approx(4.0)


def test_clip_state_to_norm_rescales_global_update_norm():
    update = OrderedDict(weight=torch.tensor([3.0, 4.0]))

    clipped, original_norm, scale = clip_state_to_norm(update, 2.5)

    assert original_norm == pytest.approx(5.0)
    assert scale == pytest.approx(0.5)
    assert state_l2_norm(clipped) == pytest.approx(2.5)


def test_adaptive_clipped_rdp_accountant_matches_paper_formula():
    accountant = AdaptiveClippedRdpAccountant(rdp_alpha=16.0, delta=1e-5, noise_multiplier=2.0)

    step = accountant.step(
        round_index=0,
        sampling_rate=1.0,
        adaptive_clip_norm=3.0,
        reference_clip_norm=6.0,
        median_update_norm=2.5,
        raw_clip_norm=3.0,
    )

    expected_round_rdp = (1.0 ** 2) * 16.0 / (2.0 * (2.0 ** 2)) * ((3.0 / 6.0) ** 2)
    expected_epsilon = expected_round_rdp + math.log(1e5) / (16.0 - 1.0)
    assert step.round_rdp == pytest.approx(expected_round_rdp)
    assert step.total_rdp == pytest.approx(expected_round_rdp)
    assert step.epsilon == pytest.approx(expected_epsilon)
    assert step.noise_std == pytest.approx(12.0)


def test_adaptive_clipped_rdp_accountant_requires_valid_parameters():
    with pytest.raises(ValueError, match="rdp_alpha"):
        AdaptiveClippedRdpAccountant(rdp_alpha=1.0, delta=1e-5, noise_multiplier=1.0)

    with pytest.raises(ValueError, match="delta"):
        AdaptiveClippedRdpAccountant(rdp_alpha=8.0, delta=1.5, noise_multiplier=1.0)
