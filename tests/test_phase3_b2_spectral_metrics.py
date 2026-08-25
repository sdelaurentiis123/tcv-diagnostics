from __future__ import annotations

import math

import numpy as np
import pytest

from tcv_diagnostics.b2_spectral_metrics import (
    B2_MODE_BANDS,
    B2SpectralAccumulator,
    derived_ensemble_calibration,
)
from tcv_diagnostics.metrics import fair_crps


def _multi_mode_truth(targets: int, shape: tuple[int, int, int]) -> np.ndarray:
    n_x, n_y, n_z = shape
    z = 2.0 * np.pi * np.arange(n_z, dtype=np.float64) / n_z
    x_factor = 1.0 + 0.05 * np.arange(n_x, dtype=np.float64)
    y_factor = 1.0 + 0.03 * np.arange(n_y, dtype=np.float64)
    spatial = x_factor[:, None, None] * y_factor[None, :, None]
    result = np.zeros((targets, 5, *shape), dtype=np.float64)
    for target in range(targets):
        for channel in range(5):
            curve = np.zeros(n_z, dtype=np.float64)
            for mode in range(1, 8):
                amplitude = (1.0 + 0.1 * channel) / (mode + 1)
                phase = 0.07 * channel * mode + 0.03 * target
                curve += amplitude * np.cos(mode * z + phase)
            result[target, channel] = (1.0 + 0.04 * target) * spatial * curve
    return result


def _assert_json_finite_or_none(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_finite_or_none(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_finite_or_none(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_derived_calibration_matches_direct_fair_crps_and_prefixes():
    generator = np.random.default_rng(17)
    truth = generator.normal(size=9)
    forecast = truth[:, None] + generator.normal(scale=0.5, size=(9, 32))
    result = derived_ensemble_calibration(
        forecast,
        truth,
        model_seed=1701,
        target_frames=tuple(range(498, 507)),
        variable_code=123,
    )
    assert result["primary_M32"]["fair_crps"] == pytest.approx(
        np.mean(fair_crps(forecast, truth, member_axis=1))
    )
    assert tuple(result["member_prefix_sensitivity"]) == (
        "M4",
        "M8",
        "M16",
        "M32",
    )
    assert result["rank_histogram"]["total"] == 9
    assert result["order_statistic_intervals"]["I31"][
        "nominal_coverage"
    ] == pytest.approx(31 / 33)


def test_memberwise_spectra_recover_identical_multimode_truth_and_n_equals_5k():
    shape = (4, 4, 16)
    targets = (498, 499)
    truth = _multi_mode_truth(len(targets), shape)
    forecast = np.repeat(truth[:, None], 32, axis=1)
    scorer = B2SpectralAccumulator(
        model_seed=1701,
        target_frames=targets,
        eligible_xy_mask=np.ones(shape[:2], dtype=bool),
        volume_shape=shape,
    )
    for index, target in enumerate(targets):
        scorer.update(
            target_frame=target,
            physical_forecast=forecast[index],
            physical_truth=truth[index],
        )
    result = scorer.finalize()
    assert result["stored_k"] == list(range(9))
    assert result["full_torus_n"] == [5 * value for value in range(9)]
    assert result["memberwise_nonlinear_diagnostic_before_ensemble_reduction"]
    assert result["ensemble_mean_fields_used_as_probabilistic_spectrum"] is False
    for field in ("Ne", "Pe", "Pi", "phi", "Vi"):
        for label, _, _ in B2_MODE_BANDS:
            band = result["toroidal_field_power"][field]["bands"][label]
            assert band["member_expected_power_ratio"] == pytest.approx(1.0)
            assert band[
                "ensemble_mean_realization_coherence_with_truth"
            ] == pytest.approx(1.0)
    for pair in ("Ne-phi", "Pe-phi", "Pi-phi"):
        for label, _, _ in B2_MODE_BANDS:
            band = result["toroidal_cross_field"][pair]["bands"][label]
            assert band[
                "truth_amplitude_weighted_absolute_phase_error_degrees"
            ] == pytest.approx(0.0, abs=1e-12)
            assert band[
                "truth_amplitude_weighted_absolute_coherence_change"
            ] == pytest.approx(0.0, abs=1e-12)
    _assert_json_finite_or_none(result)


def test_spectral_sparse_target_extension_is_explicit_and_opt_in():
    shape = (4, 4, 16)
    targets = (498, 500)
    truth = _multi_mode_truth(len(targets), shape)
    forecast = np.repeat(truth[:, None], 32, axis=1)
    with pytest.raises(ValueError, match="contiguous"):
        B2SpectralAccumulator(
            model_seed=1702,
            target_frames=targets,
            eligible_xy_mask=np.ones(shape[:2], dtype=bool),
            volume_shape=shape,
        )
    scorer = B2SpectralAccumulator(
        model_seed=1702,
        target_frames=targets,
        eligible_xy_mask=np.ones(shape[:2], dtype=bool),
        volume_shape=shape,
        allow_sparse_targets=True,
    )
    for index, target in enumerate(targets):
        scorer.update(
            target_frame=target,
            physical_forecast=forecast[index],
            physical_truth=truth[index],
        )
    result = scorer.finalize()
    assert result["target_frames"] == [498, 500]
    assert result["target_frames_are_explicit_indices"] is True


def test_spectra_are_invariant_to_independent_phi_gauge_offsets():
    shape = (4, 4, 16)
    truth = _multi_mode_truth(1, shape)[0]
    forecast = np.repeat(truth[None], 32, axis=0)
    shifted_forecast = forecast.copy()
    shifted_truth = truth.copy()
    shifted_forecast[:, 3] += np.linspace(-20.0, 40.0, 32)[:, None, None, None]
    shifted_truth[3] += 100.0
    first = B2SpectralAccumulator(
        model_seed=1701,
        target_frames=(498,),
        eligible_xy_mask=np.ones(shape[:2], dtype=bool),
        volume_shape=shape,
    )
    second = B2SpectralAccumulator(
        model_seed=1701,
        target_frames=(498,),
        eligible_xy_mask=np.ones(shape[:2], dtype=bool),
        volume_shape=shape,
    )
    first.update(
        target_frame=498,
        physical_forecast=forecast,
        physical_truth=truth,
    )
    second.update(
        target_frame=498,
        physical_forecast=shifted_forecast,
        physical_truth=shifted_truth,
    )
    baseline = first.finalize()
    shifted = second.finalize()
    for curve in (
        "truth_power",
        "member_expected_power",
        "ensemble_mean_field_power",
    ):
        np.testing.assert_allclose(
            shifted["toroidal_field_power"]["phi"]["curves"][curve][1:],
            baseline["toroidal_field_power"]["phi"]["curves"][curve][1:],
            rtol=1e-12,
            atol=1e-14,
        )
    for pair in ("Ne-phi", "Pe-phi", "Pi-phi"):
        for part in ("real", "imag"):
            np.testing.assert_allclose(
                shifted["toroidal_cross_field"][pair]["curves"][
                    "member_expected_cross_spectrum"
                ][part][1:],
                baseline["toroidal_cross_field"][pair]["curves"][
                    "member_expected_cross_spectrum"
                ][part][1:],
                rtol=1e-12,
                atol=1e-12,
            )
    for direction in ("x", "y"):
        for curve in (
            "truth_power",
            "member_expected_power",
            "ensemble_mean_field_power",
        ):
            np.testing.assert_allclose(
                shifted["directional_index_spectra"][direction]["fields"][
                    "phi"
                ][curve],
                baseline["directional_index_spectra"][direction]["fields"][
                    "phi"
                ][curve],
                rtol=1e-12,
                atol=1e-12,
            )


def test_member_expected_power_is_not_power_of_ensemble_mean():
    shape = (4, 4, 16)
    z = 2.0 * np.pi * np.arange(shape[-1]) / shape[-1]
    mode = np.cos(2.0 * z)
    truth = np.broadcast_to(mode, (5, *shape)).copy()
    forecast = np.repeat(truth[None], 32, axis=0)
    forecast[16:] *= -1.0
    scorer = B2SpectralAccumulator(
        model_seed=1701,
        target_frames=(498,),
        eligible_xy_mask=np.ones(shape[:2], dtype=bool),
        volume_shape=shape,
    )
    scorer.update(
        target_frame=498,
        physical_forecast=forecast,
        physical_truth=truth,
    )
    result = scorer.finalize()
    curves = result["toroidal_field_power"]["Ne"]["curves"]
    assert curves["member_expected_power"][2] == pytest.approx(
        curves["truth_power"][2]
    )
    assert curves["ensemble_mean_field_power"][2] == pytest.approx(0.0, abs=1e-20)
    assert curves["member_expected_power"][2] > 0.0


def test_spectral_mirror_reuses_one_transform_and_matches_standalone_block():
    shape = (4, 4, 16)
    truth = _multi_mode_truth(2, shape)
    forecast = np.repeat(truth[:, None], 32, axis=1)
    overall = B2SpectralAccumulator(
        model_seed=1701,
        target_frames=(498, 499),
        eligible_xy_mask=np.ones(shape[:2], dtype=bool),
        volume_shape=shape,
    )
    mirror = B2SpectralAccumulator(
        model_seed=1701,
        target_frames=(498,),
        eligible_xy_mask=np.ones(shape[:2], dtype=bool),
        volume_shape=shape,
    )
    standalone = B2SpectralAccumulator(
        model_seed=1701,
        target_frames=(498,),
        eligible_xy_mask=np.ones(shape[:2], dtype=bool),
        volume_shape=shape,
    )
    overall.update(
        target_frame=498,
        physical_forecast=forecast[0],
        physical_truth=truth[0],
        mirrors=(mirror,),
    )
    overall.update(
        target_frame=499,
        physical_forecast=forecast[1],
        physical_truth=truth[1],
    )
    standalone.update(
        target_frame=498,
        physical_forecast=forecast[0],
        physical_truth=truth[0],
    )
    assert mirror.finalize() == standalone.finalize()
    assert overall.finalize()["target_count"] == 2
