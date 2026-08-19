"""Known-answer tests for frozen residual-KL evaluation rules."""

from __future__ import annotations

import copy

import numpy as np

from tcv_diagnostics.b5_covariance_localization import CovarianceSummaryAccumulator
from tcv_diagnostics.b5_residual_audit import B5_FIELDS
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES
from tcv_diagnostics.residual_kl_metrics import (
    material_power_ratio_summary,
    paired_relative_l2,
    projection_dependence_distance_summary,
    projection_dependence_pass_summary,
    representation_pass_summary,
    residual_cross_spectral_summary,
    static_covariance_usefulness_summary,
)


def _covariance_record(values: np.ndarray) -> dict:
    mask = np.ones(values.shape[2:4], dtype=bool)
    accumulator = CovarianceSummaryAccumulator(
        region_masks_xy={"eligible_union": mask},
        volume_shape=values.shape[2:],
    )
    accumulator.update(values)
    record, _ = accumulator.finalize()
    return record


def test_cross_spectral_identity_has_zero_phase_and_coherence_error() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(8, 5, 2, 3, 16))
    values[:, 3] += 0.7 * values[:, 0]
    record = residual_cross_spectral_summary(
        values,
        values,
        eligible_xy_mask=np.ones((2, 3), dtype=bool),
    )
    assert record["mode_mapping"] == "n=5k"
    for pair in record["pairs"].values():
        for band in pair["bands"].values():
            assert abs(band["signed_cross_phase_error_degrees"]) < 1e-12
            assert band["absolute_coherence_error"] < 1e-14


def test_dependence_rule_counts_only_no_worse_than_training_drift() -> None:
    rng = np.random.default_rng(5)
    training_values = rng.normal(size=(12, 5, 2, 3, 16))
    validation_values = training_values + 0.05 * rng.normal(
        size=training_values.shape
    )
    training = _covariance_record(training_values)
    validation = _covariance_record(validation_values)
    blocks = {f"b{index}": validation for index in range(6)}
    projection_blocks = {f"b{index}": validation for index in range(6)}
    identities = [
        "cross_field:eligible_union",
        *[f"spatial:x:{field}" for field in B5_FIELDS],
        *[f"spatial:y:{field}" for field in B5_FIELDS],
    ]
    assert len(identities) == 11
    result = projection_dependence_pass_summary(
        training=training,
        validation_blocks=blocks,
        projection_blocks=projection_blocks,
        systematic_identities=identities,
    )
    assert result["identity_pass_count"] == 11
    assert result["passes"] is True
    assert set(result["direction_counts"].values()) == {6}


def test_all_dependence_distances_cover_every_axis_field_and_region() -> None:
    rng = np.random.default_rng(51)
    training_values = rng.normal(size=(12, 5, 2, 3, 16))
    validation_values = training_values + 0.08 * rng.normal(
        size=training_values.shape
    )
    training = _covariance_record(training_values)
    validation = _covariance_record(validation_values)
    result = projection_dependence_distance_summary(
        training=training,
        validation=validation,
        projection=validation,
    )
    assert tuple(result["spatial"]) == ("x", "y", "stored_toroidal_z")
    for axis in result["spatial"].values():
        assert tuple(axis) == B5_FIELDS
        assert all(record["projection_to_validation_RMS"] == 0.0 for record in axis.values())
    assert tuple(result["cross_field"]) == tuple(validation["cross_field"])
    assert all(
        record["projection_to_validation_RMS"] == 0.0
        for record in result["cross_field"].values()
    )


def test_material_power_rule_is_exactly_twelve_of_fifteen() -> None:
    rng = np.random.default_rng(6)
    values = rng.normal(size=(8, 5, 2, 3, 16))
    validation = _covariance_record(values)
    projection = copy.deepcopy(validation)
    result = material_power_ratio_summary(
        projection=projection,
        validation=validation,
    )
    assert result["in_range_count"] == 15
    assert result["passes"] is True
    for field in B5_FIELDS[:2]:
        for band in ("k1_3", "k4_5"):
            projection["toroidal_support"]["fields"][field]["bands"][band][
                "mean_parseval_power_density"
            ] *= 0.5
    failed = material_power_ratio_summary(
        projection=projection,
        validation=validation,
    )
    assert failed["in_range_count"] == 11
    assert failed["passes"] is False


def test_representation_gate_is_conjunctive() -> None:
    variance = {"total": 0.9, "fields": {field: 0.8 for field in B5_FIELDS}}
    dependence = {"passes": True, "identity_pass_count": 10}
    power = {"passes": True, "in_range_count": 13}
    transport = {
        quantity: {
            "strict_face_relative_L2": 0.2,
            "exact_separatrix_relative_L2": 0.2,
        }
        for quantity in TRANSPORT_QUANTITIES
    }
    passed = representation_pass_summary(
        variance_capture=variance,
        dependence=dependence,
        material_power=power,
        transport=transport,
    )
    assert passed["passes"] is True
    variance["fields"]["phi"] = 0.59
    failed = representation_pass_summary(
        variance_capture=variance,
        dependence=dependence,
        material_power=power,
        transport=transport,
    )
    assert failed["passes"] is False
    assert failed["components"]["variance"]["passes"] is False


def _transport_covariance_record(
    *, local: float, integrated: float, ratio: float, counterfactual: float
) -> dict:
    return {
        "covariance_decomposition": {
            "local_corrected_spread_skill_ratio": local,
            "integrated_corrected_spread_skill_ratio": integrated,
            "ensemble_to_error_coherence_multiplier_ratio": ratio,
            "counterfactual_local_spread_skill_after_same_factor": counterfactual,
        }
    }


def test_static_usefulness_requires_three_transport_and_field_calibration() -> None:
    quantities = {
        quantity: _transport_covariance_record(
            local=1.0,
            integrated=1.0,
            ratio=0.8,
            counterfactual=1.0,
        )
        for quantity in TRANSPORT_QUANTITIES
    }
    result = static_covariance_usefulness_summary(
        field_corrected_spread_skill=1.0,
        transport_covariance_quantities=quantities,
        finite_noncollapsed_members=True,
    )
    assert result["transport_pass_count"] == 4
    assert result["passes"] is True
    for quantity in TRANSPORT_QUANTITIES[:2]:
        quantities[quantity] = _transport_covariance_record(
            local=1.0,
            integrated=0.4,
            ratio=0.8,
            counterfactual=1.0,
        )
    failed = static_covariance_usefulness_summary(
        field_corrected_spread_skill=1.0,
        transport_covariance_quantities=quantities,
        finite_noncollapsed_members=True,
    )
    assert failed["transport_pass_count"] == 2
    assert failed["passes"] is False


def test_paired_relative_l2_known_answer() -> None:
    truth = np.asarray([3.0, 4.0])
    assert paired_relative_l2(truth, truth) == 0.0
    assert paired_relative_l2(np.zeros_like(truth), truth) == 1.0
