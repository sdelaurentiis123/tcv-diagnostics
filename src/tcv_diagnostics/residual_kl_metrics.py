"""Frozen evaluation helpers for the Paper 0 residual-KL oracle.

All functions are data-independent.  They consume already computed arrays or
compact covariance records and cannot load data, forecasts, or models.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b5_covariance_localization import (
    correlation_curve_distance,
    off_diagonal_rms_distance,
)
from .b5_residual_audit import B5_FIELDS
from .codec_transport import TRANSPORT_QUANTITIES


KL_CROSS_PAIRS = (("Ne", "phi"), ("Pe", "phi"), ("Pi", "phi"))
KL_MATERIAL_BANDS = (("k1_3", 1, 3), ("k4_5", 4, 5), ("k6_7", 6, 7))


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _canonical(name: str, values: np.ndarray) -> np.ndarray:
    array = _finite_real(name, values)
    if array.ndim != 5 or array.shape[1] != len(B5_FIELDS):
        raise ValueError(f"{name} must be [sample,field,x,y,z]")
    if array.shape[0] < 1 or min(array.shape[2:]) < 2:
        raise ValueError(f"{name} has a short dimension")
    return array


def _one_sided_weights(n_z: int) -> np.ndarray:
    size = int(n_z)
    if size < 2:
        raise ValueError("toroidal axis is too short")
    weights = np.ones(size // 2 + 1, dtype=np.float64)
    if size % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    return weights


def _json_float(value: float) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def residual_cross_spectral_summary(
    candidate: np.ndarray,
    truth: np.ndarray,
    *,
    eligible_xy_mask: np.ndarray,
    zperiod: int = 5,
) -> dict[str, Any]:
    """Compare residual density/pressure--phi cross-phase and coherence."""

    forecast = _canonical("cross-spectral candidate", candidate)
    observed = _canonical("cross-spectral truth", truth)
    if forecast.shape != observed.shape:
        raise ValueError("cross-spectral candidate and truth shapes differ")
    if int(zperiod) != 5:
        raise ValueError("residual KL cross spectra require zperiod=5")
    mask = np.asarray(eligible_xy_mask, dtype=bool)
    if mask.shape != forecast.shape[2:4] or not np.any(mask):
        raise ValueError("cross-spectral eligible mask differs")
    candidate_fft = np.fft.rfft(forecast, axis=-1)[:, :, mask, :]
    truth_fft = np.fft.rfft(observed, axis=-1)[:, :, mask, :]
    weights = _one_sided_weights(forecast.shape[-1])
    records: dict[str, Any] = {}
    for first_name, second_name in KL_CROSS_PAIRS:
        first = B5_FIELDS.index(first_name)
        second = B5_FIELDS.index(second_name)

        def spectra(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            auto_first = np.sum(
                np.abs(values[:, first]) ** 2,
                axis=(0, 1),
                dtype=np.float64,
            )
            auto_second = np.sum(
                np.abs(values[:, second]) ** 2,
                axis=(0, 1),
                dtype=np.float64,
            )
            cross = np.sum(
                values[:, first] * np.conjugate(values[:, second]),
                axis=(0, 1),
                dtype=np.complex128,
            )
            return auto_first, auto_second, cross

        truth_a, truth_b, truth_cross = spectra(truth_fft)
        candidate_a, candidate_b, candidate_cross = spectra(candidate_fft)
        bands: dict[str, Any] = {}
        for label, low, high in KL_MATERIAL_BANDS:
            indices = slice(low, high + 1)
            band_weights = weights[indices]
            truth_complex = np.sum(truth_cross[indices] * band_weights)
            candidate_complex = np.sum(candidate_cross[indices] * band_weights)
            truth_auto_a = float(np.sum(truth_a[indices] * band_weights))
            truth_auto_b = float(np.sum(truth_b[indices] * band_weights))
            candidate_auto_a = float(np.sum(candidate_a[indices] * band_weights))
            candidate_auto_b = float(np.sum(candidate_b[indices] * band_weights))
            truth_coherence = (
                float(np.abs(truth_complex) ** 2 / (truth_auto_a * truth_auto_b))
                if truth_auto_a > 0.0 and truth_auto_b > 0.0
                else math.nan
            )
            candidate_coherence = (
                float(
                    np.abs(candidate_complex) ** 2
                    / (candidate_auto_a * candidate_auto_b)
                )
                if candidate_auto_a > 0.0 and candidate_auto_b > 0.0
                else math.nan
            )
            phase_error = (
                math.degrees(
                    float(
                        np.angle(candidate_complex * np.conjugate(truth_complex))
                    )
                )
                if np.abs(candidate_complex) > 0.0 and np.abs(truth_complex) > 0.0
                else math.nan
            )
            bands[label] = {
                "stored_k_inclusive": [low, high],
                "full_torus_n_inclusive": [zperiod * low, zperiod * high],
                "truth_cross_real": float(np.real(truth_complex)),
                "truth_cross_imaginary": float(np.imag(truth_complex)),
                "candidate_cross_real": float(np.real(candidate_complex)),
                "candidate_cross_imaginary": float(np.imag(candidate_complex)),
                "signed_cross_phase_error_degrees": _json_float(phase_error),
                "truth_coherence": _json_float(truth_coherence),
                "candidate_coherence": _json_float(candidate_coherence),
                "absolute_coherence_error": _json_float(
                    abs(candidate_coherence - truth_coherence)
                ),
            }
        records[f"{first_name}-{second_name}"] = {"bands": bands}
    return {
        "target_count": int(forecast.shape[0]),
        "eligible_xy_cells": int(np.sum(mask)),
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "pairs": records,
    }


def projection_dependence_distance_summary(
    *,
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare every frozen ACF curve and regional cross-field matrix.

    ``training`` is the training-only covariance reference constructed from
    the same centered matrix used to fit the KL basis.  The empirical
    training-to-validation distance is a drift reference, not a sampling
    distribution or confidence interval.
    """

    spatial: dict[str, Any] = {}
    for axis in ("x", "y", "stored_toroidal_z"):
        axis_records: dict[str, Any] = {}
        for field in B5_FIELDS:
            train_curve = training["spatial_autocorrelation"][axis]["fields"][
                field
            ]["correlation"]
            validation_curve = validation["spatial_autocorrelation"][axis][
                "fields"
            ][field]["correlation"]
            projection_curve = projection["spatial_autocorrelation"][axis][
                "fields"
            ][field]["correlation"]
            drift = correlation_curve_distance(train_curve, validation_curve)
            distance = correlation_curve_distance(projection_curve, validation_curve)
            axis_records[field] = {
                "training_to_validation_RMS": drift,
                "projection_to_validation_RMS": distance,
                "projection_no_worse_than_training_drift": distance <= drift,
            }
        spatial[axis] = axis_records

    # JSON persistence sorts mapping keys, whereas freshly accumulated records
    # retain the frozen B2 insertion order.  Region names are the semantic
    # identifiers; serialization order is not.  Require exact name equality,
    # then emit results in the validation record's canonical in-memory order.
    regions = tuple(validation["cross_field"])
    region_names = frozenset(regions)
    if any(
        frozenset(record["cross_field"]) != region_names
        for record in (training, projection)
    ):
        raise ValueError("KL dependence-distance cross-field region names differ")
    cross_field: dict[str, Any] = {}
    for region in regions:
        validation_matrix = np.asarray(
            validation["cross_field"][region]["correlation_matrix"]
        )
        drift = off_diagonal_rms_distance(
            np.asarray(training["cross_field"][region]["correlation_matrix"]),
            validation_matrix,
        )
        distance = off_diagonal_rms_distance(
            np.asarray(projection["cross_field"][region]["correlation_matrix"]),
            validation_matrix,
        )
        cross_field[region] = {
            "training_to_validation_RMS": drift,
            "projection_to_validation_RMS": distance,
            "projection_no_worse_than_training_drift": distance <= drift,
        }
    return {
        "spatial": spatial,
        "cross_field": cross_field,
        "empirical_training_to_validation_drift_is_not_a_sampling_distribution": True,
    }


def projection_dependence_pass_summary(
    *,
    training: Mapping[str, Any],
    validation_blocks: Mapping[str, Mapping[str, Any]],
    projection_blocks: Mapping[str, Mapping[str, Any]],
    systematic_identities: Sequence[str],
) -> dict[str, Any]:
    """Apply the frozen nine-of-eleven and five-of-six dependence rule."""

    names = tuple(validation_blocks)
    if len(names) != 6 or tuple(projection_blocks) != names:
        raise ValueError("KL dependence rule requires six ordered blocks")
    identities = tuple(str(value) for value in systematic_identities)
    if len(identities) != 11 or len(set(identities)) != 11:
        raise ValueError("KL dependence rule requires eleven unique identities")
    counts = {identity: 0 for identity in identities}
    blocks: dict[str, Any] = {}
    for name in names:
        validation = validation_blocks[name]
        projection = projection_blocks[name]
        records: dict[str, Any] = {}
        for identity in identities:
            kind, middle, *tail = identity.split(":")
            if kind == "spatial" and len(tail) == 1:
                axis = middle
                field = tail[0]
                train_curve = training["spatial_autocorrelation"][axis]["fields"][
                    field
                ]["correlation"]
                validation_curve = validation["spatial_autocorrelation"][axis][
                    "fields"
                ][field]["correlation"]
                projection_curve = projection["spatial_autocorrelation"][axis][
                    "fields"
                ][field]["correlation"]
                drift = correlation_curve_distance(train_curve, validation_curve)
                distance = correlation_curve_distance(
                    projection_curve, validation_curve
                )
            elif kind == "cross_field" and not tail:
                region = middle
                validation_matrix = np.asarray(
                    validation["cross_field"][region]["correlation_matrix"]
                )
                drift = off_diagonal_rms_distance(
                    np.asarray(
                        training["cross_field"][region]["correlation_matrix"]
                    ),
                    validation_matrix,
                )
                distance = off_diagonal_rms_distance(
                    np.asarray(
                        projection["cross_field"][region]["correlation_matrix"]
                    ),
                    validation_matrix,
                )
            else:
                raise ValueError(f"unrecognized dependence identity {identity!r}")
            passes = distance <= drift
            counts[identity] += int(passes)
            records[identity] = {
                "training_to_validation_drift": drift,
                "projection_to_validation_distance": distance,
                "projection_no_worse_than_drift": passes,
            }
        blocks[name] = records
    passed = sorted(identity for identity, count in counts.items() if count >= 5)
    return {
        "chronological_block_count": 6,
        "required_blocks_per_identity": 5,
        "systematic_identity_total": 11,
        "required_systematic_identities": 9,
        "direction_counts": counts,
        "identities_passing_five_of_six": passed,
        "identity_pass_count": len(passed),
        "passes": len(passed) >= 9,
        "by_block": blocks,
    }


def material_power_ratio_summary(
    *,
    projection: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen 12-of-15 absolute residual-power rule."""

    ratios: dict[str, Any] = {}
    count = 0
    for field in B5_FIELDS:
        field_record: dict[str, Any] = {}
        for label, low, high in KL_MATERIAL_BANDS:
            projected = float(
                projection["toroidal_support"]["fields"][field]["bands"][label][
                    "mean_parseval_power_density"
                ]
            )
            truth = float(
                validation["toroidal_support"]["fields"][field]["bands"][label][
                    "mean_parseval_power_density"
                ]
            )
            ratio = projected / truth if truth > 0.0 else math.nan
            passes = math.isfinite(ratio) and 0.8 <= ratio <= 1.2
            count += int(passes)
            field_record[label] = {
                "stored_k_inclusive": [low, high],
                "full_torus_n_inclusive": [5 * low, 5 * high],
                "projection_power": projected,
                "validation_power": truth,
                "power_ratio": _json_float(ratio),
                "in_frozen_range": passes,
            }
        ratios[field] = field_record
    return {
        "material_field_band_total": 15,
        "required_in_range": 12,
        "power_ratio_range": [0.8, 1.2],
        "in_range_count": count,
        "passes": count >= 12,
        "fields": ratios,
    }


def paired_relative_l2(candidate: np.ndarray, truth: np.ndarray) -> float:
    forecast = _finite_real("paired candidate", candidate)
    observed = _finite_real("paired truth", truth)
    if forecast.shape != observed.shape:
        raise ValueError("paired relative-L2 shapes differ")
    denominator = float(np.sum(observed * observed, dtype=np.float64))
    if denominator <= 0.0:
        return math.nan
    return math.sqrt(
        float(np.sum((forecast - observed) ** 2, dtype=np.float64)) / denominator
    )


def representation_pass_summary(
    *,
    variance_capture: Mapping[str, Any],
    dependence: Mapping[str, Any],
    material_power: Mapping[str, Any],
    transport: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Apply the conjunctive frozen Tier-A representation gate."""

    if tuple(transport) != TRANSPORT_QUANTITIES:
        raise ValueError("Tier-A transport quantity order differs")
    field_capture = variance_capture.get("fields", {})
    if tuple(field_capture) != B5_FIELDS:
        raise ValueError("Tier-A variance-capture field order differs")
    variance_pass = float(variance_capture["total"]) >= 0.80 and all(
        float(field_capture[field]) >= 0.60 for field in B5_FIELDS
    )
    strict_pass_count = sum(
        math.isfinite(float(record["strict_face_relative_L2"]))
        and float(record["strict_face_relative_L2"]) <= 0.40
        for record in transport.values()
    )
    exact_pass_count = sum(
        math.isfinite(float(record["exact_separatrix_relative_L2"]))
        and float(record["exact_separatrix_relative_L2"]) <= 0.30
        for record in transport.values()
    )
    transport_pass = strict_pass_count >= 3 and exact_pass_count == 4
    components = {
        "variance": {
            "passes": variance_pass,
            "minimum_total": 0.80,
            "minimum_each_field": 0.60,
        },
        "dependence": {
            "passes": bool(dependence["passes"]),
            "identity_pass_count": int(dependence["identity_pass_count"]),
            "required": 9,
        },
        "material_power": {
            "passes": bool(material_power["passes"]),
            "in_range_count": int(material_power["in_range_count"]),
            "required": 12,
        },
        "transport": {
            "passes": transport_pass,
            "strict_face_pass_count": strict_pass_count,
            "strict_face_required": 3,
            "exact_separatrix_pass_count": exact_pass_count,
            "exact_separatrix_required": 4,
        },
    }
    return {
        "conjunctive_all_components_required": True,
        "components": components,
        "passes": all(record["passes"] for record in components.values()),
    }


def static_covariance_usefulness_summary(
    *,
    field_corrected_spread_skill: float,
    transport_covariance_quantities: Mapping[str, Mapping[str, Any]],
    finite_noncollapsed_members: bool,
) -> dict[str, Any]:
    """Apply the frozen Tier-B static-covariance usefulness rule."""

    if tuple(transport_covariance_quantities) != TRANSPORT_QUANTITIES:
        raise ValueError("Tier-B transport quantity order differs")
    records: dict[str, Any] = {}
    passing = 0
    for quantity, record in transport_covariance_quantities.items():
        values = record["covariance_decomposition"]
        local = float(values["local_corrected_spread_skill_ratio"])
        integrated = float(values["integrated_corrected_spread_skill_ratio"])
        ratio = float(values["ensemble_to_error_coherence_multiplier_ratio"])
        counterfactual = float(
            values["counterfactual_local_spread_skill_after_same_factor"]
        )
        checks = {
            "local_spread_skill": math.isfinite(local) and 0.80 <= local <= 1.25,
            "integrated_spread_skill": (
                math.isfinite(integrated) and 0.67 <= integrated <= 1.50
            ),
            "coherence_multiplier_ratio": math.isfinite(ratio) and ratio >= 0.67,
            "scalar_counterfactual_local": (
                math.isfinite(counterfactual) and counterfactual <= 1.50
            ),
        }
        passed = all(checks.values())
        passing += int(passed)
        records[quantity] = {
            "values": {
                "local_corrected_spread_skill_ratio": local,
                "integrated_corrected_spread_skill_ratio": integrated,
                "ensemble_to_error_coherence_multiplier_ratio": ratio,
                "counterfactual_local_spread_skill_after_same_factor": counterfactual,
            },
            "checks": checks,
            "passes": passed,
        }
    field_ratio = float(field_corrected_spread_skill)
    field_pass = math.isfinite(field_ratio) and 0.80 <= field_ratio <= 1.25
    return {
        "transport_quantities": records,
        "transport_pass_count": passing,
        "transport_required": 3,
        "aggregate_field_corrected_spread_skill_ratio": field_ratio,
        "aggregate_field_passes": field_pass,
        "finite_noncollapsed_members": bool(finite_noncollapsed_members),
        "passes": (
            passing >= 3 and field_pass and bool(finite_noncollapsed_members)
        ),
        "paper0_forecast_acceptance_gate": False,
    }
