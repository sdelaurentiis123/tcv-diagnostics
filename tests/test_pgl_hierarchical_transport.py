from __future__ import annotations

import math

import numpy as np
import torch

from tcv_diagnostics.pgl_hierarchical_transport import (
    PGL_HIERARCHICAL_LOW_K,
    PGL_HIERARCHICAL_LOW_N,
    PGL_HIERARCHICAL_TRANSPORT_K,
    PGL_HIERARCHICAL_TRANSPORT_N,
    fair_crps_score,
    fair_energy_score,
    global_sum_from_k0,
    global_transport_sum,
    regional_transport_sums,
    score_hierarchical_transport,
    toroidal_transport_coefficients,
)
from tcv_diagnostics.pgl_variogram import IndexedPairBank, fair_variogram_score


def _one_pair_bank() -> IndexedPairBank:
    return IndexedPairBank(
        left=np.asarray([0], dtype=np.int64),
        right=np.asarray([1], dtype=np.int64),
        weight=np.asarray([1.0], dtype=np.float64),
        group=np.asarray([0], dtype=np.int64),
        group_name="known_answer",
        group_values=(1.0,),
        metadata={"test": True},
    )


def test_fair_energy_known_answers_and_gradient() -> None:
    truth = torch.tensor([[0.0, 1.0, -2.0]])
    exact = truth[:, None].repeat(1, 4, 1)
    result = fair_energy_score(exact, truth)
    assert result.fair.item() == 0.0
    assert result.ordinary.item() == 0.0

    wrong = (exact + torch.tensor([0.3, -0.2, 0.7])).requires_grad_(True)
    score = fair_energy_score(wrong, truth)
    assert score.fair.item() > 0.0
    score.fair.backward()
    assert wrong.grad is not None
    assert torch.isfinite(wrong.grad).all()
    assert torch.count_nonzero(wrong.grad) > 0
    assert torch.allclose(
        score.ordinary - score.fair,
        score.finite_member_correction,
        atol=1.0e-7,
        rtol=0.0,
    )


def test_fair_crps_known_answers_and_gradient() -> None:
    truth = torch.tensor([[0.0, 1.0, -2.0]])
    exact = truth[:, None].repeat(1, 4, 1)
    assert fair_crps_score(exact, truth).fair.item() == 0.0

    biased = (exact + 0.5).requires_grad_(True)
    score = fair_crps_score(biased, truth)
    assert math.isclose(score.fair.item(), 0.5, abs_tol=1.0e-7)
    score.fair.backward()
    assert biased.grad is not None
    assert torch.isfinite(biased.grad).all()
    assert torch.count_nonzero(biased.grad) > 0


def test_regions_are_disjoint_complete_and_preserve_global_sum() -> None:
    values = torch.arange(16 * 81, dtype=torch.float64).reshape(16, 81)
    regions = regional_transport_sums(values)
    assert regions.shape == (12,)
    assert torch.equal(regions.sum(), values.sum())

    markers = torch.zeros((16, 81), dtype=torch.float32)
    for poloidal in range(4):
        for toroidal in range(3):
            markers[poloidal * 4 : (poloidal + 1) * 4, toroidal * 27 : (toroidal + 1) * 27] = (
                10 * poloidal + toroidal + 1
            )
    observed = regional_transport_sums(markers).reshape(4, 3)
    expected = torch.tensor(
        [[108.0 * (10 * poloidal + toroidal + 1) for toroidal in range(3)] for poloidal in range(4)]
    )
    assert torch.equal(observed, expected)


def test_fourier_zero_mode_equals_global_and_mode_mapping_is_physical() -> None:
    generator = torch.Generator().manual_seed(85604)
    local = torch.randn((2, 4, 16, 81), generator=generator)
    assert torch.allclose(
        global_sum_from_k0(local), global_transport_sum(local), atol=2.0e-5, rtol=2.0e-6
    )
    assert PGL_HIERARCHICAL_LOW_N == tuple(5 * k for k in PGL_HIERARCHICAL_LOW_K)
    assert PGL_HIERARCHICAL_TRANSPORT_N == tuple(
        5 * k for k in PGL_HIERARCHICAL_TRANSPORT_K
    )

    z = torch.arange(81, dtype=torch.float32)
    profile = torch.cos(2.0 * math.pi * 4.0 * z / 81.0)
    manufactured = profile[None].expand(16, -1) / 16.0
    spectrum = toroidal_transport_coefficients(manufactured)
    assert int(torch.argmax(torch.abs(spectrum[1:])).item()) + 1 == 4


def test_local_variogram_misses_common_mode_but_hierarchy_detects_it() -> None:
    truth = torch.zeros((1, 4, 4, 16, 81), dtype=torch.float32)
    members = truth[:, None].repeat(1, 4, 1, 1, 1, 1)
    spatial_bank = _one_pair_bank()
    exact_local = fair_variogram_score(
        members[:, :, :, :1], truth[:, :, :1], spatial_bank
    )
    shifted = members + 2.0
    shifted_local = fair_variogram_score(
        shifted[:, :, :, :1], truth[:, :, :1], spatial_bank
    )
    assert exact_local.fair.item() == 0.0
    assert shifted_local.fair.item() == 0.0
    shifted_total = global_transport_sum(shifted)
    truth_total = global_transport_sum(truth)
    assert fair_crps_score(shifted_total, truth_total).fair.item() > 0.0
    assert fair_energy_score(
        regional_transport_sums(shifted), regional_transport_sums(truth)
    ).fair.item() > 0.0


def test_full_hierarchy_truth_like_scores_are_zero() -> None:
    generator = torch.Generator().manual_seed(1702)
    truth = torch.randn((1, 4, 4, 16, 81), generator=generator)
    members = truth[:, None].repeat(1, 4, 1, 1, 1, 1)
    current = torch.randn((1, 1, 4, 16, 81), generator=generator)
    trajectory_truth = torch.cat((current, truth), dim=1)
    trajectory_members = torch.cat(
        (current[:, None].expand(1, 4, 1, 4, 16, 81), members), dim=2
    )
    scores = score_hierarchical_transport(
        local_members=members,
        local_future_truth=truth,
        local_trajectory_members=trajectory_members,
        local_trajectory_truth=trajectory_truth,
        spatial_bank=_one_pair_bank(),
        temporal_bank=_one_pair_bank(),
    )
    for values in (
        scores.local_spatial,
        scores.local_temporal,
        scores.regional,
        scores.fourier_low,
        scores.fourier_transport_band,
        scores.global_crps,
    ):
        assert all(abs(float(value)) <= 1.0e-7 for value in values)
