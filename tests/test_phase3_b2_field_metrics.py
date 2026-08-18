from __future__ import annotations

import numpy as np
import pytest

from tcv_diagnostics.b2_field_metrics import (
    B2_ALL_REGIONS,
    B2_INTERVALS,
    FieldRegionAccumulator,
    b2_region_masks,
    gauge_fix_phi_channel,
    pointwise_ensemble_diagnostics,
)
from tcv_diagnostics.b2_probabilistic_metrics import (
    order_statistic_interval_coverage,
)
from tcv_diagnostics.geometry import SingleNullRegionMasks
from tcv_diagnostics.metrics import fair_crps, ordinary_crps


def _synthetic_regions() -> SingleNullRegionMasks:
    shape = (64, 32)
    strict = np.ones(shape, dtype=bool)
    operator = np.zeros(shape, dtype=bool)
    operator[:, 1:-1] = True
    eligible = strict & operator
    confined = np.zeros(shape, dtype=bool)
    confined[:32, 10:21] = True
    private = np.zeros(shape, dtype=bool)
    private[:32] = eligible[:32] & ~confined[:32]
    sol = np.zeros(shape, dtype=bool)
    sol[32:] = eligible[32:]
    separatrix = np.zeros(shape, dtype=bool)
    separatrix[31:33, 1:-1] = True
    outboard = np.zeros(shape, dtype=bool)
    outboard[:, 15] = True
    xpoint = np.zeros(shape, dtype=bool)
    xpoint[31:33, [5, 10, 20, 25]] = True
    inner = np.zeros(shape, dtype=bool)
    inner[:, 1:6] = True
    outer = np.zeros(shape, dtype=bool)
    outer[:, 25:31] = True
    return SingleNullRegionMasks(
        strict_wall_interior=strict,
        wall_crossing=np.zeros(shape, dtype=bool),
        wall_exterior=np.zeros(shape, dtype=bool),
        operator_interior=operator,
        confined_edge=confined,
        private_flux=private,
        scrape_off_layer=sol,
        separatrix_cell_band=separatrix,
        outboard_midplane=outboard,
        x_point_topology_stencil=xpoint,
        inner_divertor_leg=inner,
        outer_divertor_leg=outer,
        separatrix_x_index=32,
        separatrix_face_left_cell_index=31,
        core_lower_y=10,
        core_upper_y=20,
        outboard_midplane_y=15,
    )


def test_b2_region_masks_use_authoritative_partition_and_expand_z():
    regions = b2_region_masks(_synthetic_regions(), n_z=3)
    assert tuple(regions) == B2_ALL_REGIONS
    assert all(mask.shape == (64 * 32 * 3,) for mask in regions.values())
    primary_sum = sum(
        regions[name].astype(np.int8)
        for name in ("confined_edge", "private_flux", "scrape_off_layer")
    )
    np.testing.assert_array_equal(primary_sum == 1, regions["eligible_union"])
    assert not np.any(primary_sum > 1)
    assert int(np.sum(regions["eligible_union"])) == 64 * 30 * 3


def test_phi_gauge_fix_removes_independent_member_and_truth_offsets():
    generator = np.random.default_rng(4)
    base_forecast = generator.normal(size=(4, 2, 3, 5))
    base_truth = generator.normal(size=(2, 3, 5))
    member_offsets = np.asarray([3.0, -7.0, 11.0, 0.5])[:, None, None, None]
    fixed_forecast, fixed_truth = gauge_fix_phi_channel(
        base_forecast + member_offsets,
        base_truth + 19.0,
    )
    expected_forecast, expected_truth = gauge_fix_phi_channel(
        base_forecast,
        base_truth,
    )
    np.testing.assert_allclose(fixed_forecast, expected_forecast, atol=2e-15)
    np.testing.assert_allclose(fixed_truth, expected_truth, atol=2e-15)
    np.testing.assert_allclose(
        np.mean(fixed_forecast, axis=(1, 2, 3)), 0.0, atol=3e-15
    )
    assert float(np.mean(fixed_truth)) == pytest.approx(0.0, abs=3e-15)


def test_pointwise_diagnostics_match_independent_crps_and_interval_oracles():
    generator = np.random.default_rng(1701)
    forecast = generator.normal(size=(32, 11))
    truth = generator.normal(size=11)
    diagnostics = pointwise_ensemble_diagnostics(
        forecast,
        truth,
        target_frame=498,
        channel_index=2,
        spatial_cell_index=np.arange(11),
        tie_seed=123,
    )
    np.testing.assert_allclose(
        diagnostics.fair_crps,
        fair_crps(forecast, truth, member_axis=0),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostics.ordinary_crps,
        ordinary_crps(forecast, truth, member_axis=0),
        rtol=1e-14,
        atol=1e-14,
    )
    for name, (lower, upper) in B2_INTERVALS.items():
        expected = order_statistic_interval_coverage(
            forecast,
            truth,
            lower_order_one_indexed=lower,
            upper_order_one_indexed=upper,
            member_axis=0,
        )
        np.testing.assert_array_equal(
            diagnostics.interval_covered[name], expected["covered"]
        )
        np.testing.assert_allclose(
            diagnostics.interval_width[name],
            expected["upper"] - expected["lower"],
        )


def test_pointwise_tie_ranks_are_stateless_and_reproducible():
    truth = np.asarray([0.0, 1.0, 2.0])
    forecast = np.tile(truth, (32, 1))
    first = pointwise_ensemble_diagnostics(
        forecast,
        truth,
        target_frame=500,
        channel_index=3,
        spatial_cell_index=np.asarray([9, 10, 11]),
        tie_seed=987,
    )
    repeated = pointwise_ensemble_diagnostics(
        forecast,
        truth,
        target_frame=500,
        channel_index=3,
        spatial_cell_index=np.asarray([9, 10, 11]),
        tie_seed=987,
    )
    assert np.all(first.tied)
    np.testing.assert_array_equal(first.ranks, repeated.ranks)
    assert np.all((first.ranks >= 0) & (first.ranks <= 32))


def test_field_region_accumulator_reduces_pointwise_arrays_without_voxel_claim():
    generator = np.random.default_rng(12)
    forecast = generator.normal(size=(32, 17))
    truth = generator.normal(size=17)
    diagnostics = pointwise_ensemble_diagnostics(
        forecast,
        truth,
        target_frame=501,
        channel_index=0,
        spatial_cell_index=np.arange(17),
        tie_seed=2468,
    )
    mask = np.asarray([True] * 9 + [False] * 8)
    accumulator = FieldRegionAccumulator()
    accumulator.update(diagnostics, truth, mask)
    result = accumulator.finalize()

    mean = np.mean(forecast[:, mask], axis=0)
    assert result["scalar_count"] == 9
    assert result["voxel_count_used_as_independent_sample_size"] is False
    assert result["ensemble_mean"]["rmse"] == pytest.approx(
        np.sqrt(np.mean((mean - truth[mask]) ** 2))
    )
    assert result["fair_crps"] == pytest.approx(
        np.mean(fair_crps(forecast[:, mask], truth[mask], member_axis=0))
    )
    expected_spread = np.sqrt(
        (33 / 32) * np.mean(np.var(forecast[:, mask], axis=0, ddof=1))
    )
    assert result["corrected_spread_skill"]["corrected_rms_spread"] == pytest.approx(
        expected_spread
    )
    assert sum(result["rank_histogram"]["counts"]) == 9
    assert result["rank_histogram"]["pixel_iid_p_value_reported"] is False
    assert result["spread_integrity"]["nonzero_spread"] is True
