"""Synthetic-oracle tests for B5 covariance-localization primitives."""

from __future__ import annotations

import math

import numpy as np

from tcv_diagnostics.b5_covariance_localization import (
    B5_FINITE_MEMBER_FACTOR,
    CrossFieldCorrelationAccumulator,
    SpatialCorrelationAccumulator,
    ToroidalPowerAccumulator,
    TransportCovarianceAccumulator,
    association_summary,
    axisymmetric_bias,
    classify_localization,
    correlation_curve_distance,
    deterministic_field_error_summary,
    deterministic_toroidal_summary,
    field_variogram_score,
    gauge_fix_fields,
    off_diagonal_rms_distance,
    subtract_axisymmetric_bias,
    training_frozen_ar1_coefficients,
    training_frozen_ar1_prediction,
    transport_variogram_score,
)
from tcv_diagnostics.b5_residual_audit import spatial_autocorrelation


def random_fields(
    samples: int,
    shape: tuple[int, int, int] = (6, 5, 12),
    seed: int = 41,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(samples, 5, *shape)).astype(np.float32)


def test_gauge_fix_and_axisymmetric_bias_have_exact_semantics() -> None:
    values = random_fields(4)
    original = values.copy()
    fixed = gauge_fix_fields(values)
    assert np.array_equal(fixed[:, :3], original[:, :3])
    assert np.array_equal(fixed[:, 4], original[:, 4])
    assert np.allclose(np.mean(fixed[:, 3], axis=(1, 2, 3)), 0.0, atol=2e-7)
    assert np.array_equal(values, original)
    bias = axisymmetric_bias(fixed)
    fluctuation = subtract_axisymmetric_bias(fixed, bias)
    assert bias.shape == (5, 6, 5)
    assert np.allclose(np.mean(fluctuation, axis=(0, 4)), 0.0, atol=2e-7)


def test_streamed_spatial_correlations_match_frozen_batch_estimator() -> None:
    values = random_fields(7)
    bias = axisymmetric_bias(values)
    fluctuation = subtract_axisymmetric_bias(values, bias)
    for axis in ("x", "y", "stored_toroidal_z"):
        expected, _ = spatial_autocorrelation(fluctuation, axis=axis)
        accumulator = SpatialCorrelationAccumulator(
            axis=axis, volume_shape=values.shape[2:]
        )
        accumulator.update(fluctuation[:3])
        accumulator.update(fluctuation[3:])
        actual, raw = accumulator.finalize()
        assert actual["sample_count"] == 7
        for field in ("Ne", "Pe", "Pi", "phi", "Vi"):
            assert np.allclose(
                actual["fields"][field]["correlation"],
                expected["fields"][field]["correlation"],
                atol=2e-7,
            )
        assert np.array_equal(raw["lags"], np.arange(raw["correlation"].shape[1]))
    assert correlation_curve_distance([1.0, 0.5, 0.0], [1.0, 0.3, 0.2]) == 0.2


def test_cross_field_accumulator_recovers_known_dependence_and_distance() -> None:
    generator = np.random.default_rng(9)
    base = generator.normal(size=(10, 4, 3, 8)).astype(np.float32)
    values = np.stack(
        [base, 2.0 * base, -base, generator.normal(size=base.shape), 0.5 * base],
        axis=1,
    )
    masks = {
        "left": np.pad(np.ones((2, 3), dtype=bool), ((0, 2), (0, 0))),
        "right": np.pad(np.ones((2, 3), dtype=bool), ((2, 0), (0, 0))),
    }
    accumulator = CrossFieldCorrelationAccumulator(
        region_masks_xy=masks, volume_shape=(4, 3, 8)
    )
    accumulator.update(values[:4])
    accumulator.update(values[4:])
    record, raw = accumulator.finalize()
    correlation = np.asarray(record["global"]["correlation_matrix"])
    assert math.isclose(correlation[0, 1], 1.0, abs_tol=1e-7)
    assert math.isclose(correlation[0, 2], -1.0, abs_tol=1e-7)
    assert math.isclose(correlation[0, 4], 1.0, abs_tol=1e-7)
    assert record["global"]["sample_count"] == 10 * 4 * 3 * 8
    assert raw["left__centered_gram"].shape == (5, 5)
    assert off_diagonal_rms_distance(correlation, correlation) == 0.0


def test_toroidal_power_localizes_a_known_k6_mode_to_n30() -> None:
    n_z = 88
    z = np.arange(n_z)
    wave = np.sin(2.0 * np.pi * 6.0 * z / n_z).astype(np.float32)
    values = np.broadcast_to(wave, (3, 5, 4, 3, n_z)).copy()
    accumulator = ToroidalPowerAccumulator(volume_shape=(4, 3, n_z), sample_chunk=2)
    accumulator.update(values[:1])
    accumulator.update(values[1:])
    record, raw = accumulator.finalize()
    for field in record["fields"].values():
        band = field["bands"]["k6_7"]
        assert band["full_torus_n_inclusive"] == [30, 35]
        assert band["power_fraction"] > 0.999999
    assert int(np.argmax(raw["mean_parseval_power_density"][0])) == 6


def test_variogram_scores_are_zero_for_an_exact_ensemble() -> None:
    truth = random_fields(1, shape=(4, 3, 88), seed=3)[0]
    forecast = np.repeat(truth[None], 6, axis=0)
    masks = {"eligible": np.ones((4, 3), dtype=bool)}
    field_score = field_variogram_score(forecast, truth, region_masks_xy=masks)
    assert field_score == {"global": 0.0, "eligible": 0.0}
    local_truth = truth[0, :2, 0, :]
    local_members = np.repeat(local_truth[None], 6, axis=0)
    transport_score = transport_variogram_score(local_members, local_truth)
    assert transport_score["equal_lag_mean"] == 0.0
    assert set(transport_score["by_lag"]) == {
        "lag_1",
        "lag_2",
        "lag_4",
        "lag_8",
        "lag_16",
        "lag_32",
        "lag_40",
    }


def test_transport_covariance_decomposes_diagonal_and_coherent_variance() -> None:
    members = 32
    rows = 2
    n_z = 81
    generator = np.random.default_rng(17)
    amplitudes = generator.normal(size=members)
    pattern = np.ones((rows, n_z), dtype=np.float64)
    forecast_values = amplitudes[:, None, None] * pattern[None]
    truth_values = np.full((rows, n_z), 0.25)
    quantities = ("particle", "electron_internal_energy", "ion_internal_energy", "total_internal_energy")
    accumulator = TransportCovarianceAccumulator(
        quantities=quantities, rows=rows, n_z=n_z
    )
    for target in (498, 499, 500):
        forecast = {name: forecast_values + 0.01 * (target - 498) for name in quantities}
        truth = {name: truth_values for name in quantities}
        accumulator.update(target_frame=target, forecast=forecast, truth=truth)
    record, raw = accumulator.finalize()
    metrics = record["quantities"]["particle"]["covariance_decomposition"]
    point_count = rows * n_z
    assert metrics["target_count"] == 3
    assert metrics["local_point_count_per_target"] == point_count
    assert math.isclose(
        metrics["ensemble_coherence_multiplier"], point_count, rel_tol=1e-12
    )
    assert metrics["member_variance_ddof"] == 1
    assert metrics["finite_member_variance_factor"] == B5_FINITE_MEMBER_FACTOR
    assert raw["particle__innovation_local_error"].shape == (3, rows, n_z)
    assert record["nonlinear_operator_applied_memberwise_before_reduction"] is True


def test_training_frozen_ar1_uses_only_supplied_training_sufficient_statistics() -> None:
    numerator = np.zeros((5, 4), dtype=np.float64)
    left_energy = np.ones((5, 4), dtype=np.float64)
    numerator[:, 1] = np.asarray([-0.2, -0.1, 0.0, 0.25, 0.5])
    raw = {
        "temporal_pattern__numerator": numerator,
        "temporal_pattern__left_energy": left_energy,
    }
    coefficients = training_frozen_ar1_coefficients(raw)
    assert np.array_equal(coefficients, numerator[:, 1])
    mean = np.zeros((2, 5, 3, 2, 4), dtype=np.float32)
    previous = np.ones_like(mean)
    bias = np.full((5, 3, 2), 2.0)
    prediction = training_frozen_ar1_prediction(
        h1_mean=mean,
        previous_h1_residual=previous,
        coefficients=coefficients,
        axisymmetric_training_bias=bias,
    )
    expected = bias[None, ..., None] + coefficients[None, :, None, None, None] * (
        1.0 - bias[None, ..., None]
    )
    assert np.allclose(prediction, expected)


def test_deterministic_metrics_remove_phi_gauge_and_resolve_known_mode() -> None:
    truth = np.zeros((3, 5, 4, 3, 88), dtype=np.float32)
    prediction = truth.copy()
    prediction[:, 3] = 9.0
    summary = deterministic_field_error_summary(prediction, truth)
    assert summary["equal_field_mean_RMSE"] == 0.0
    z = np.arange(88)
    truth[:, :, :, :, :] = np.sin(2 * np.pi * 4 * z / 88)
    prediction = truth.copy()
    spectral = deterministic_toroidal_summary(prediction, truth)
    for field in spectral["fields"].values():
        assert math.isclose(field["bands"]["k4_5"]["power_ratio"], 1.0)
        assert math.isclose(
            field["bands"]["k4_5"]["realization_coherence"], 1.0
        )


def test_association_and_frozen_interpretation_labels_are_explicit() -> None:
    association = association_summary([1, 2, 3, 4], [2, 4, 6, 8])
    assert math.isclose(association["pearson"], 1.0)
    assert math.isclose(association["spearman"], 1.0)
    assert association["calibration_proof"] is False
    quantities = {}
    for index, name in enumerate(("a", "b", "c", "d")):
        quantities[name] = {
            "covariance_decomposition": {
                "local_corrected_spread_skill_ratio": 1.0 if index < 3 else 0.5,
                "integrated_corrected_spread_skill_ratio": 0.5,
                "ensemble_to_error_coherence_multiplier_ratio": 0.5,
                "counterfactual_local_spread_skill_after_same_factor": 2.0,
            }
        }
    labels = classify_localization(
        transport_quantities=quantities,
        history_aggregate_improvement_fraction=0.03,
        history_improved_block_count=5,
    )
    assert labels["L1_predominantly_amplitude_limited"]["supported"] is False
    assert labels["L2_covariance_organization_limited"]["supported"] is True
    assert labels["L4_explicit_residual_history_signal"]["supported"] is True
    assert labels["training_authorized"] is False
    assert labels["held_out_85606_access_authorized"] is False
