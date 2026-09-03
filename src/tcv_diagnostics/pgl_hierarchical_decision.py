"""Fixed decision logic for the two-arm hierarchical transport screen."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping

from .pgl_hierarchical_training import (
    PGL_HIERARCHICAL_ARMS,
    PGL_HIERARCHICAL_CHECKPOINT_UPDATES,
)


def extract_gate_metrics(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the preregistered scientific metrics from one M32 gate."""

    family = gate.get("family_pass", {})
    required_families = (
        "field_distribution",
        "spectral_retention",
        "cross_field",
        "integrated_transport_mean",
        "local_transport_calibration",
        "integrated_transport_calibration",
        "spatial_transport_covariance",
    )
    if set(family) != set(required_families):
        raise ValueError("hierarchical gate families differ")
    local = gate["local_transport_calibration"]["corrected_spread_skill_by_quantity"]
    values = [float(value) for value in local.values()]
    record = {
        "production_passed": bool(gate["passed"]),
        "family_pass": {name: bool(family[name]) for name in required_families},
        "integrated_spread_skill": float(
            gate["integrated_transport_calibration"]["median_corrected_spread_skill"]
        ),
        "spatial_covariance_error": float(
            gate["spatial_transport_covariance"]["median_relative_frobenius_error"]
        ),
        "local_spread_skill_median": float(median(values)),
        "local_spread_skill_by_quantity": {
            str(name): float(value) for name, value in local.items()
        },
        "mean_transport_relative_l2": float(
            gate["integrated_transport_mean"]["candidate_median_relative_L2"]
        ),
        "spectral_error": float(
            gate["spectral_retention"][
                "candidate_median_absolute_log_power_ratio_error"
            ]
        ),
        "cross_spectrum_error": float(
            gate["cross_field"]["candidate"][
                "normalized_complex_cross_spectrum_error_k1_7"
            ]
        ),
        "phase_error_degrees": float(
            gate["cross_field"]["candidate"][
                "truth_amplitude_weighted_absolute_phase_error_degrees_k1_7"
            ]
        ),
        "field_fair_crps_h4": float(
            gate["field_distribution"]["h4"]["candidate_equal_field_fair_CRPS"]
        ),
    }
    numeric = [
        value
        for name, value in record.items()
        if name not in ("production_passed", "family_pass", "local_spread_skill_by_quantity")
    ] + list(record["local_spread_skill_by_quantity"].values())
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("hierarchical gate metric is non-finite")
    return record


def evaluate_two_epoch_decision(
    records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen update-428 continuation and production gates."""

    expected = {
        (arm, update)
        for arm in PGL_HIERARCHICAL_ARMS
        for update in PGL_HIERARCHICAL_CHECKPOINT_UPDATES
    }
    if set(records) != expected:
        raise ValueError("hierarchical decision requires all six fixed evaluations")
    metrics = {
        key: extract_gate_metrics(record["gate"]) for key, record in records.items()
    }
    control = metrics[("CONTROL", 428)]
    treatment = metrics[("TRANSPORT", 428)]
    integrated_gain = (
        treatment["integrated_spread_skill"] - control["integrated_spread_skill"]
    )
    covariance_reduction = (
        control["spatial_covariance_error"] - treatment["spatial_covariance_error"]
    )
    stability_families = (
        "field_distribution",
        "spectral_retention",
        "cross_field",
        "integrated_transport_mean",
    )
    stable = all(treatment["family_pass"][name] for name in stability_families)
    extension = bool(
        integrated_gain >= 0.05 and covariance_reduction >= 0.01 and stable
    )
    production = bool(treatment["production_passed"])
    return {
        "schema_version": 1,
        "scope": "old_85604_pgl_hierarchical_two_epoch_decision",
        "metrics": {
            f"{arm}_update_{update}": metrics[(arm, update)]
            for arm in PGL_HIERARCHICAL_ARMS
            for update in PGL_HIERARCHICAL_CHECKPOINT_UPDATES
        },
        "epoch_two_matched_difference": {
            "integrated_spread_skill_gain": integrated_gain,
            "minimum_required": 0.05,
            "spatial_covariance_error_reduction": covariance_reduction,
            "minimum_required_covariance_reduction": 0.01,
            "mean_phase_spectra_field_gates_stable": stable,
        },
        "longer_4_to_8_epoch_extension_authorized": extension,
        "transport_arm_passed_production_gate": production,
        "next_action": (
            "advance_to_confirmation_seeds_without_extending_duration"
            if production
            else "write_dated_longer_duration_amendment"
            if extension
            else "stop_hierarchical_transport_training"
        ),
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
