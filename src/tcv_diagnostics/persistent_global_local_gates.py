"""Prospectively frozen gate reductions for the persistent global--local pilot."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS
from .codec_transport import TRANSPORT_QUANTITIES


PGL_SPECTRAL_BANDS = ("k1_3", "k4_5", "k6_7")


def _finite_scalar(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _complex_curve(record: Mapping[str, Sequence[float]], label: str) -> np.ndarray:
    real = np.asarray(record.get("real"), dtype=np.float64)
    imag = np.asarray(record.get("imag"), dtype=np.float64)
    if real.shape != imag.shape or real.ndim != 1 or real.size < 8:
        raise ValueError(f"{label} complex curve differs")
    values = real + 1j * imag
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} complex curve is non-finite")
    return values


def _field_gate(
    candidate: Mapping[str, Any], selected_mean: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_union = candidate["regions"]["eligible_union"]
    selected_union = selected_mean["regions"]["eligible_union"]
    fair = _finite_scalar(
        candidate_union["aggregate"]["equal_channel_fair_crps"],
        "candidate equal-field fair CRPS",
    )
    deterministic_mae = _finite_scalar(
        selected_union["aggregate"]["equal_channel_ensemble_mean_mae"],
        "selected-mean equal-field MAE",
    )
    per_field = {
        field: _finite_scalar(
            candidate_union["fields"][field]["corrected_spread_skill"]["ratio"],
            f"{field} spread-skill",
        )
        for field in B2_FIELDS
    }
    return {
        "candidate_equal_field_fair_CRPS": fair,
        "selected_mean_equal_field_MAE": deterministic_mae,
        "fair_CRPS_strictly_below_selected_mean_MAE": fair < deterministic_mae,
        "per_field_corrected_spread_skill": per_field,
        "maximum_per_field_corrected_spread_skill": max(per_field.values()),
        "maximum_allowed": 1.5,
        "no_field_overdispersed_above_1p50": max(per_field.values()) <= 1.5,
    }


def _spectral_gate(
    candidate: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_errors = []
    parent_errors = []
    records = []
    for field in B2_FIELDS:
        for band in PGL_SPECTRAL_BANDS:
            candidate_ratio = _finite_scalar(
                candidate["toroidal_field_power"][field]["bands"][band][
                    "member_expected_power_ratio"
                ],
                f"candidate {field} {band} power ratio",
            )
            parent_ratio = _finite_scalar(
                parent["toroidal_field_power"][field]["bands"][band][
                    "member_expected_power_ratio"
                ],
                f"parent {field} {band} power ratio",
            )
            if candidate_ratio <= 0.0 or parent_ratio <= 0.0:
                raise ValueError("spectral gate power ratio must be positive")
            candidate_error = abs(math.log(candidate_ratio))
            parent_error = abs(math.log(parent_ratio))
            candidate_errors.append(candidate_error)
            parent_errors.append(parent_error)
            records.append(
                {
                    "field": field,
                    "band": band,
                    "candidate_power_ratio": candidate_ratio,
                    "parent_power_ratio": parent_ratio,
                    "candidate_absolute_log_error": candidate_error,
                    "parent_absolute_log_error": parent_error,
                }
            )
    candidate_median = float(np.median(candidate_errors))
    parent_median = float(np.median(parent_errors))
    relative = candidate_median / parent_median if parent_median > 0.0 else math.inf
    return {
        "matched_field_band_records": records,
        "candidate_median_absolute_log_power_ratio_error": candidate_median,
        "parent_median_absolute_log_power_ratio_error": parent_median,
        "candidate_over_parent": relative,
        "maximum_allowed": 1.1,
        "passes": math.isfinite(relative) and relative <= 1.1,
    }


def _cross_field_summary(record: Mapping[str, Any]) -> dict[str, float]:
    curves = record["toroidal_cross_field"]["Ne-phi"]["curves"]
    truth = _complex_curve(curves["truth_cross_spectrum"], "truth Ne-phi")
    expected = _complex_curve(
        curves["member_expected_cross_spectrum"], "forecast Ne-phi"
    )
    selected = slice(1, 8)
    truth_selected = truth[selected]
    expected_selected = expected[selected]
    denominator = float(np.linalg.norm(truth_selected))
    normalized_error = (
        float(np.linalg.norm(expected_selected - truth_selected)) / denominator
        if denominator > 0.0
        else math.inf
    )
    weights = np.abs(truth_selected)
    weight_sum = float(np.sum(weights))
    phase = np.abs(np.angle(expected_selected * np.conjugate(truth_selected)))
    phase_degrees = (
        math.degrees(float(np.sum(weights * phase) / weight_sum))
        if weight_sum > 0.0
        else math.inf
    )
    truth_coherence = np.asarray(curves["truth_coherence"], dtype=np.float64)[selected]
    expected_coherence = np.asarray(
        curves["member_expected_coherence"], dtype=np.float64
    )[selected]
    coherence_error = (
        float(np.sum(weights * np.abs(expected_coherence - truth_coherence)) / weight_sum)
        if weight_sum > 0.0
        else math.inf
    )
    if not all(math.isfinite(value) for value in (normalized_error, phase_degrees, coherence_error)):
        raise ValueError("Ne-phi gate summary is non-finite")
    return {
        "normalized_complex_cross_spectrum_error_k1_7": normalized_error,
        "truth_amplitude_weighted_absolute_phase_error_degrees_k1_7": phase_degrees,
        "truth_amplitude_weighted_absolute_coherence_error_k1_7": coherence_error,
    }


def _transport_relative_l2(record: Mapping[str, Any], quantity: str) -> float:
    return _finite_scalar(
        record["quantities"][quantity]["reductions"]["separatrix_wedge"][
            "ensemble_expected_paired_metrics"
        ]["relative_l2"],
        f"{quantity} integrated transport relative L2",
    )


def evaluate_pgl_physics_gates(
    *,
    candidate_h1_field: Mapping[str, Any],
    candidate_h4_field: Mapping[str, Any],
    selected_h1_field: Mapping[str, Any],
    selected_h4_field: Mapping[str, Any],
    candidate_h4_spectral: Mapping[str, Any],
    parent_h4_spectral: Mapping[str, Any],
    candidate_h4_transport: Mapping[str, Any],
    parent_h4_transport: Mapping[str, Any],
    candidate_h4_covariance: Mapping[str, Any],
    candidate_h4_spatial_sketch: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply every frozen gate without an aggregate compensation score."""

    field_h1 = _field_gate(candidate_h1_field, selected_h1_field)
    field_h4 = _field_gate(candidate_h4_field, selected_h4_field)
    field_pass = all(
        record["fair_CRPS_strictly_below_selected_mean_MAE"]
        and record["no_field_overdispersed_above_1p50"]
        for record in (field_h1, field_h4)
    )
    spectral = _spectral_gate(candidate_h4_spectral, parent_h4_spectral)
    candidate_cross = _cross_field_summary(candidate_h4_spectral)
    parent_cross = _cross_field_summary(parent_h4_spectral)
    cross = {
        "candidate": candidate_cross,
        "parent": parent_cross,
        "complex_error_strictly_improves": (
            candidate_cross["normalized_complex_cross_spectrum_error_k1_7"]
            < parent_cross["normalized_complex_cross_spectrum_error_k1_7"]
        ),
        "phase_error_increase_degrees": (
            candidate_cross[
                "truth_amplitude_weighted_absolute_phase_error_degrees_k1_7"
            ]
            - parent_cross[
                "truth_amplitude_weighted_absolute_phase_error_degrees_k1_7"
            ]
        ),
        "maximum_phase_error_increase_degrees": 2.0,
    }
    cross["passes"] = bool(
        cross["complex_error_strictly_improves"]
        and cross["phase_error_increase_degrees"] <= 2.0
    )

    sketch_values = {
        quantity: _finite_scalar(
            candidate_h4_spatial_sketch["quantities"][quantity][
                "relative_frobenius_error_sketch"
            ],
            f"{quantity} spatial covariance error",
        )
        for quantity in TRANSPORT_QUANTITIES
    }
    sketch_median = float(np.median(list(sketch_values.values())))
    spatial = {
        "relative_frobenius_error_by_quantity": sketch_values,
        "median_relative_frobenius_error": sketch_median,
        "maximum_allowed": 0.9,
        "passes": sketch_median < 0.9,
    }

    covariance_quantities = candidate_h4_covariance["quantities"]
    local = {
        quantity: _finite_scalar(
            covariance_quantities[quantity]["covariance_decomposition"][
                "local_corrected_spread_skill_ratio"
            ],
            f"{quantity} local spread-skill",
        )
        for quantity in TRANSPORT_QUANTITIES
    }
    local_in_interval = sum(0.8 <= value <= 1.25 for value in local.values())
    local_gate = {
        "corrected_spread_skill_by_quantity": local,
        "calibrated_interval": [0.8, 1.25],
        "quantities_in_interval": local_in_interval,
        "minimum_quantities_in_interval": 3,
        "maximum_value": max(local.values()),
        "maximum_any_allowed": 1.4,
        "passes": local_in_interval >= 3 and max(local.values()) <= 1.4,
    }
    integrated_spread = {
        quantity: _finite_scalar(
            covariance_quantities[quantity]["covariance_decomposition"][
                "integrated_corrected_spread_skill_ratio"
            ],
            f"{quantity} integrated spread-skill",
        )
        for quantity in TRANSPORT_QUANTITIES
    }
    integrated_spread_median = float(np.median(list(integrated_spread.values())))
    integrated_calibration = {
        "corrected_spread_skill_by_quantity": integrated_spread,
        "median_corrected_spread_skill": integrated_spread_median,
        "minimum_required": 0.6,
        "passes": integrated_spread_median >= 0.6,
    }
    candidate_l2 = {
        quantity: _transport_relative_l2(candidate_h4_transport, quantity)
        for quantity in TRANSPORT_QUANTITIES
    }
    parent_l2 = {
        quantity: _transport_relative_l2(parent_h4_transport, quantity)
        for quantity in TRANSPORT_QUANTITIES
    }
    candidate_median_l2 = float(np.median(list(candidate_l2.values())))
    parent_median_l2 = float(np.median(list(parent_l2.values())))
    integrated_ratio = (
        candidate_median_l2 / parent_median_l2
        if parent_median_l2 > 0.0
        else math.inf
    )
    integrated_mean = {
        "candidate_relative_L2_by_quantity": candidate_l2,
        "parent_relative_L2_by_quantity": parent_l2,
        "candidate_median_relative_L2": candidate_median_l2,
        "parent_median_relative_L2": parent_median_l2,
        "candidate_over_parent": integrated_ratio,
        "maximum_allowed": 1.05,
        "passes": math.isfinite(integrated_ratio) and integrated_ratio <= 1.05,
    }
    families = {
        "field_distribution": field_pass,
        "spectral_retention": bool(spectral["passes"]),
        "cross_field": bool(cross["passes"]),
        "spatial_transport_covariance": bool(spatial["passes"]),
        "local_transport_calibration": bool(local_gate["passes"]),
        "integrated_transport_calibration": bool(integrated_calibration["passes"]),
        "integrated_transport_mean": bool(integrated_mean["passes"]),
    }
    return {
        "schema_version": 1,
        "decision_rule": "all_seven_families_required_no_compensation",
        "field_distribution": {"h1": field_h1, "h4": field_h4, "passes": field_pass},
        "spectral_retention": spectral,
        "cross_field": cross,
        "spatial_transport_covariance": spatial,
        "local_transport_calibration": local_gate,
        "integrated_transport_calibration": integrated_calibration,
        "integrated_transport_mean": integrated_mean,
        "family_pass": families,
        "passed": all(families.values()),
    }
