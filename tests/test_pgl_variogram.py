from __future__ import annotations

import math

import numpy as np
import torch

from tcv_diagnostics.models.persistent_global_local import (
    PersistentGlobalLocalConfig,
    PersistentGlobalLocalEDM,
    PersistentNoiseConfig,
)
from tcv_diagnostics.pgl_variogram import (
    IndexedPairBank,
    build_spatial_pair_bank,
    build_temporal_pair_bank,
    differentiable_sample_normalized,
    fair_variogram_score,
    gauge_fix_phi,
    minimum_wedge_distance,
)


def two_group_bank() -> IndexedPairBank:
    return IndexedPairBank(
        left=np.asarray([0, 1, 0, 2], dtype=np.int64),
        right=np.asarray([1, 2, 2, 3], dtype=np.int64),
        weight=np.full(4, 0.25, dtype=np.float64),
        group=np.asarray([0, 0, 1, 1], dtype=np.int64),
        group_name="test",
        group_values=(1.0, 2.0),
        metadata={"scope": "synthetic"},
    )


def test_fair_variogram_known_answers_and_bias_invariance() -> None:
    truth = torch.tensor([[0.0, 1.0, 3.0, 6.0]])
    exact = truth[:, None].expand(1, 4, 4).clone()
    exact_score = fair_variogram_score(exact, truth, two_group_bank())
    assert torch.equal(exact_score.fair, torch.zeros_like(exact_score.fair))
    assert torch.equal(exact_score.ordinary, torch.zeros_like(exact_score.ordinary))

    common_bias = exact + 17.0
    biased = fair_variogram_score(common_bias, truth, two_group_bank())
    assert torch.equal(biased.fair, exact_score.fair)
    assert torch.equal(biased.ordinary, exact_score.ordinary)

    shuffled = exact[..., torch.tensor([2, 0, 3, 1])]
    shuffled_score = fair_variogram_score(shuffled, truth, two_group_bank())
    assert shuffled_score.ordinary > exact_score.ordinary


def test_fair_correction_identity_and_gradient() -> None:
    truth = torch.tensor([[0.0, 1.0, 3.0, 6.0]])
    members = torch.stack(
        [truth[0], truth[0] + torch.tensor([0.0, 0.2, -0.1, 0.3]),
         truth[0] + torch.tensor([0.2, -0.1, 0.3, 0.0]),
         truth[0] + torch.tensor([-0.2, 0.3, 0.0, -0.1])]
    )[None].requires_grad_(True)
    score = fair_variogram_score(members, truth, two_group_bank())
    assert torch.allclose(
        score.ordinary - score.fair,
        score.finite_member_correction,
        atol=2e-7,
        rtol=0.0,
    )
    score.fair.backward()
    assert members.grad is not None
    assert torch.isfinite(members.grad).all()
    assert torch.count_nonzero(members.grad) > 0


def test_minimum_wedge_distance_wraps_only_periodic_angle() -> None:
    wedge = 2.0 * math.pi / 5.0
    first = np.asarray([[1.0, 0.0, 0.01]])
    second = np.asarray([[1.0, 0.0, wedge - 0.01]])
    observed = minimum_wedge_distance(first, second)[0]
    expected = math.sqrt(2.0 - 2.0 * math.cos(0.02))
    assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
    displaced = second.copy()
    displaced[0, 1] = 0.5
    assert minimum_wedge_distance(first, displaced)[0] > observed


def test_pair_builders_balance_groups() -> None:
    wedge = 2.0 * math.pi / 5.0
    radius = np.linspace(0.7, 1.1, 64)
    zed = np.linspace(-0.5, 0.5, 64)
    phi = np.linspace(0.0, wedge, 64, endpoint=False)
    positions = np.stack((radius, zed, phi), axis=1)
    eligible = np.arange(64, dtype=np.int64)
    spatial = build_spatial_pair_bank(
        positions,
        eligible,
        future_times=2,
        variables=2,
        bins=3,
        pairs_per_bin=8,
        seed=9,
    )
    totals = np.bincount(spatial.group, weights=spatial.weight, minlength=3)
    assert np.allclose(totals, np.full(3, 1.0 / 3.0), atol=1e-12)
    assert len(spatial.sha256) == 64

    temporal = build_temporal_pair_bank(
        eligible,
        cells=64,
        trajectory_times=5,
        variables=2,
        lags=(1, 2, 3, 4),
        pairs_per_time_variable=8,
        seed=10,
    )
    temporal_totals = np.bincount(
        temporal.group, weights=temporal.weight, minlength=4
    )
    assert np.allclose(temporal_totals, np.full(4, 0.25), atol=1e-12)


def test_gauge_fix_phi_is_independent_by_member_and_time() -> None:
    values = torch.randn(2, 3, 4, 5, 2, 2, 4)
    fixed = gauge_fix_phi(values)
    assert torch.allclose(
        fixed[:, :, :, 3].mean(dim=(-3, -2, -1)),
        torch.zeros(2, 3, 4),
        atol=2e-7,
        rtol=0.0,
    )
    assert torch.equal(fixed[:, :, :, :3], values[:, :, :, :3])


def tiny_edm() -> PersistentGlobalLocalEDM:
    config = PersistentGlobalLocalConfig(
        horizon=4,
        fields=5,
        base_channels=4,
        channel_multipliers=(1,),
        residual_blocks_per_resolution=1,
        global_channels=4,
        global_pool_xy=(2, 2),
        low_mode_maximum=1,
        noise_embedding_features=16,
        group_norm_maximum_groups=4,
        kernel_size=3,
    )
    return PersistentGlobalLocalEDM(
        config,
        residual_scales=torch.ones(4, 5),
        noise_config=PersistentNoiseConfig(
            global_pool_xy=(2, 2), low_mode_maximum=1
        ),
    )


def test_differentiable_sampler_matches_frozen_forward_and_backpropagates() -> None:
    torch.manual_seed(12)
    model = tiny_edm().eval()
    current = torch.randn(1, 5, 4, 4, 8)
    mean = torch.randn(1, 4, 5, 4, 4, 8)
    initial = torch.randn(1, 2, 4, 5, 4, 4, 8)
    with torch.no_grad():
        reference = model.sample_normalized(
            current,
            mean,
            initial,
            steps=3,
            sigma_max=2.0,
            sigma_min=0.05,
            rho=3.0,
        )
    observed = differentiable_sample_normalized(
        model,
        current,
        mean,
        initial,
        steps=3,
        sigma_max=2.0,
        sigma_min=0.05,
        rho=3.0,
        activation_checkpointing=True,
    )
    assert torch.allclose(observed, reference, atol=2e-6, rtol=0.0)
    observed.square().mean().backward()
    gradients = [value.grad for value in model.parameters() if value.requires_grad]
    assert any(value is not None and torch.count_nonzero(value) for value in gradients)
    assert all(value is None or torch.isfinite(value).all() for value in gradients)
