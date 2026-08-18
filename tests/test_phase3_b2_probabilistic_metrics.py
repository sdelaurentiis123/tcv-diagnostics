"""Known-answer tests for the frozen B2 finite-ensemble conventions."""

from __future__ import annotations

import numpy as np
import pytest

from tcv_diagnostics.b2_probabilistic_metrics import (
    corrected_spread_skill_summary,
    deterministic_tie_uniform,
    ensemble_rank_histogram,
    member_prefix_views,
    monte_carlo_stability,
    moving_block_bootstrap_indices,
    order_statistic_interval_coverage,
)


def test_corrected_spread_skill_matches_exchangeable_finite_member_formula() -> None:
    forecast = np.asarray(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    truth = np.asarray([np.sqrt(3.0), -np.sqrt(3.0)])
    summary = corrected_spread_skill_summary(forecast, truth, member_axis=1)
    assert summary["ensemble_size"] == 2
    assert summary["member_variance_ddof"] == 1
    assert summary["mean_unbiased_member_variance"] == pytest.approx(2.0)
    assert summary["finite_member_variance_factor"] == pytest.approx(1.5)
    assert summary["corrected_rms_spread"] == pytest.approx(np.sqrt(3.0))
    assert summary["rmse_of_ensemble_mean"] == pytest.approx(np.sqrt(3.0))
    assert summary["spread_skill_ratio"] == pytest.approx(1.0)


def test_corrected_spread_skill_rejects_one_member() -> None:
    with pytest.raises(ValueError, match="at least two"):
        corrected_spread_skill_summary(
            np.asarray([[1.0]]), np.asarray([1.0]), member_axis=1
        )


def test_m32_intervals_use_exact_order_statistics_without_interpolation() -> None:
    members = np.arange(1.0, 33.0)
    forecast = np.broadcast_to(members[None], (4, 32))
    truth = np.asarray([8.0, 25.0, 7.9, 25.1])
    interval = order_statistic_interval_coverage(
        forecast,
        truth,
        lower_order_one_indexed=8,
        upper_order_one_indexed=25,
        member_axis=1,
    )
    assert interval["nominal_coverage"] == pytest.approx(17.0 / 33.0)
    assert interval["empirical_coverage"] == pytest.approx(0.5)
    assert interval["mean_interval_width"] == pytest.approx(17.0)
    np.testing.assert_array_equal(interval["covered"], [True, True, False, False])
    np.testing.assert_array_equal(interval["lower"], np.full(4, 8.0))
    np.testing.assert_array_equal(interval["upper"], np.full(4, 25.0))
    assert interval["quantile_method"] == "exact_order_statistics_no_interpolation"


@pytest.mark.parametrize(
    ("lower", "upper", "nominal"),
    [(8, 25, 17 / 33), (3, 30, 27 / 33), (1, 32, 31 / 33)],
)
def test_frozen_m32_interval_nominals(lower: int, upper: int, nominal: float) -> None:
    forecast = np.arange(32.0)[None]
    record = order_statistic_interval_coverage(
        forecast,
        np.asarray([15.0]),
        lower_order_one_indexed=lower,
        upper_order_one_indexed=upper,
        member_axis=1,
    )
    assert record["nominal_coverage"] == pytest.approx(nominal)


def test_rank_histogram_inserts_truth_uniformly_among_exact_ties() -> None:
    forecast = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
    )
    truth = np.asarray([1.0, 1.0, 1.0, 3.0])
    uniforms = np.asarray([0.0, np.nextafter(1.0, 0.0), 0.5, 0.0])
    record = ensemble_rank_histogram(
        forecast,
        truth,
        member_axis=1,
        tie_uniform=uniforms,
        return_ranks=True,
    )
    np.testing.assert_array_equal(record["ranks"], [1, 2, 1, 3])
    np.testing.assert_array_equal(record["counts"], [0, 2, 1, 1])
    assert record["bins"] == 4
    assert record["total"] == 4
    assert record["tied_truth_values"] == 3
    assert record["pixel_iid_p_value_reported"] is False

    with pytest.raises(ValueError, match="tie_uniform"):
        ensemble_rank_histogram(forecast, truth, member_axis=1)


def test_stateless_tie_uniform_is_exact_repeatable_and_key_sensitive() -> None:
    target = np.asarray([498, 498, 499, 499])
    channel = np.asarray([0, 1, 0, 1])
    cell = np.asarray([0, 0, 180223, 180223])
    first = deterministic_tie_uniform(
        target, channel, cell, seed=85604032
    )
    second = deterministic_tie_uniform(
        target, channel, cell, seed=85604032
    )
    np.testing.assert_array_equal(first, second)
    assert [value.hex() for value in first] == [
        "0x1.164b8135d6adcp-1",
        "0x1.94df36bf15fd0p-1",
        "0x1.87e15a588adeap-2",
        "0x1.cf86141a102f5p-1",
    ]
    assert np.all((first >= 0.0) & (first < 1.0))
    changed = deterministic_tie_uniform(
        target, channel, cell, seed=85604033
    )
    assert not np.array_equal(first, changed)


def test_member_sensitivity_uses_views_of_one_stored_member_order() -> None:
    forecast = np.arange(1 * 32 * 3, dtype=np.float64).reshape(1, 32, 3)
    views = member_prefix_views(forecast, (4, 8, 16, 32), member_axis=1)
    assert list(views) == [4, 8, 16, 32]
    for count, view in views.items():
        assert view.shape == (1, count, 3)
        np.testing.assert_array_equal(view, forecast[:, :count])
        assert np.shares_memory(view, forecast)
    with pytest.raises(ValueError, match="strictly increasing"):
        member_prefix_views(forecast, (8, 4), member_axis=1)


def test_m16_m32_stability_rule_has_frozen_scale_and_floor() -> None:
    passing = monte_carlo_stability(1.1, 1.0)
    assert passing["absolute_difference"] == pytest.approx(0.1)
    assert passing["tolerance"] == pytest.approx(0.10000001)
    assert passing["passes"] is True
    failing = monte_carlo_stability(1.10000002, 1.0)
    assert failing["passes"] is False
    near_zero = monte_carlo_stability(1.0e-8, 0.0)
    assert near_zero["passes"] is True


def test_frozen_moving_block_bootstrap_indices_are_exact_and_contiguous() -> None:
    indices = moving_block_bootstrap_indices(
        126,
        block_length=21,
        replicates=2,
        seed=85604032,
        blocks_per_replicate=6,
    )
    assert indices.shape == (2, 126)
    np.testing.assert_array_equal(
        indices[:, ::21],
        np.asarray(
            [
                [88, 78, 49, 70, 100, 92],
                [20, 59, 63, 15, 35, 13],
            ]
        ),
    )
    blocks = indices.reshape(2, 6, 21)
    np.testing.assert_array_equal(np.diff(blocks, axis=2), np.ones((2, 6, 20)))
    assert int(np.min(indices)) >= 0
    assert int(np.max(indices)) < 126


def test_bootstrap_rejects_too_few_blocks_to_rebuild_series() -> None:
    with pytest.raises(ValueError, match="too few"):
        moving_block_bootstrap_indices(
            126,
            block_length=21,
            replicates=1,
            seed=85604032,
            blocks_per_replicate=5,
        )
