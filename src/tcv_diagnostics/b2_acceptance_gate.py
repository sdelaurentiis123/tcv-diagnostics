"""Prospectively frozen B2 seed and architecture acceptance evaluation.

The thresholds come from the committed B2 evaluation manifest.  This module
contains no model inference, file access, threshold fitting, or held-out data
access.  It only reduces already verified 85604 score records.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS, B2_PRIMARY_REGIONS
from .b2_spectral_metrics import B2_CROSS_PAIRS, B2_MODE_BANDS
from .codec_transport import TRANSPORT_QUANTITIES


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    first = _finite(numerator)
    second = _finite(denominator)
    if first is None or second is None or second <= 0.0:
        return None
    return first / second


class CheckBook:
    """Collect explicit numerical and integrity checks without short circuiting."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def _numeric(
        self,
        name: str,
        value: Any,
        *,
        operator: str,
        lower: float | None = None,
        upper: float | None = None,
        median_eligible: bool = True,
    ) -> bool:
        number = _finite(value)
        passed = False
        if number is not None:
            if operator == "finite":
                passed = True
            elif operator == "<=":
                passed = number <= float(upper)
            elif operator == "<":
                passed = number < float(upper)
            elif operator == ">=":
                passed = number >= float(lower)
            elif operator == "range":
                passed = float(lower) <= number <= float(upper)
            else:
                raise ValueError(f"unsupported B2 gate operator {operator!r}")
        self.checks.append(
            {
                "name": name,
                "kind": "numeric",
                "value": number,
                "finite": number is not None,
                "operator": operator,
                "lower": lower,
                "upper": upper,
                "median_eligible": bool(median_eligible),
                "passes": bool(passed),
            }
        )
        return bool(passed)

    def le(self, name: str, value: Any, upper: float, **kwargs: Any) -> bool:
        return self._numeric(name, value, operator="<=", upper=upper, **kwargs)

    def lt(self, name: str, value: Any, upper: float, **kwargs: Any) -> bool:
        return self._numeric(name, value, operator="<", upper=upper, **kwargs)

    def ge(self, name: str, value: Any, lower: float, **kwargs: Any) -> bool:
        return self._numeric(name, value, operator=">=", lower=lower, **kwargs)

    def between(
        self,
        name: str,
        value: Any,
        lower: float,
        upper: float,
        **kwargs: Any,
    ) -> bool:
        return self._numeric(
            name,
            value,
            operator="range",
            lower=lower,
            upper=upper,
            **kwargs,
        )

    def boolean(self, name: str, value: Any) -> bool:
        passed = value is True
        self.checks.append(
            {
                "name": name,
                "kind": "integrity",
                "value": value,
                "median_eligible": False,
                "passes": passed,
            }
        )
        return passed

    def finite(self, name: str, value: Any, *, median_eligible: bool = False) -> bool:
        return self._numeric(
            name,
            value,
            operator="finite",
            median_eligible=median_eligible,
        )

    @property
    def passes(self) -> bool:
        return bool(self.checks) and all(item["passes"] for item in self.checks)

    def record(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "check_count": len(self.checks),
            "failed_check_count": sum(not item["passes"] for item in self.checks),
            "checks": self.checks,
        }


def _coverage_error(record: Mapping[str, Any], interval: str) -> float | None:
    item = record.get("order_statistic_intervals", {}).get(interval, {})
    empirical = _finite(item.get("empirical_coverage"))
    nominal = _finite(item.get("nominal_coverage"))
    if empirical is None or nominal is None:
        return None
    return abs(empirical - nominal)


def _mc_fraction(record: Mapping[str, Any]) -> float | None:
    difference = _finite(record.get("absolute_difference"))
    tolerance = _finite(record.get("tolerance"))
    if difference is None or tolerance is None or tolerance <= 0.0:
        return None
    return difference / tolerance


def _field_scope_checks(
    probabilistic: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    uncompressed: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    prefix: str,
    median_eligible: bool,
) -> CheckBook:
    book = CheckBook()
    aggregate = probabilistic.get("aggregate", {})
    deterministic_aggregate = deterministic
    uncompressed_aggregate = uncompressed
    book.le(
        f"{prefix}.aggregate_rmse_relative_to_paired_H2",
        _ratio(
            aggregate.get("equal_channel_ensemble_mean_rmse"),
            deterministic_aggregate.get("aggregate_equal_channel_rmse_standardized"),
        ),
        float(gates["aggregate_mean_rmse_relative_to_paired_deterministic_max"]),
        median_eligible=median_eligible,
    )
    book.le(
        f"{prefix}.aggregate_mae_relative_to_paired_H2",
        _ratio(
            aggregate.get("equal_channel_ensemble_mean_mae"),
            deterministic_aggregate.get("aggregate_equal_channel_mae_standardized"),
        ),
        float(gates["aggregate_mean_mae_relative_to_paired_deterministic_max"]),
        median_eligible=median_eligible,
    )
    book.lt(
        f"{prefix}.aggregate_fCRPS_relative_to_paired_H2_MAE",
        _ratio(
            aggregate.get("equal_channel_fair_crps"),
            deterministic_aggregate.get("aggregate_equal_channel_mae_standardized"),
        ),
        1.0,
        median_eligible=median_eligible,
    )
    book.lt(
        f"{prefix}.aggregate_fCRPS_relative_to_best_uncompressed_MAE",
        _ratio(
            aggregate.get("equal_channel_fair_crps"),
            uncompressed_aggregate.get("aggregate_equal_channel_mae_standardized"),
        ),
        1.0,
        median_eligible=median_eligible,
    )

    fields = probabilistic.get("fields", {})
    deterministic_fields = deterministic.get("fields", {})
    fcrps_ratios: list[float | None] = []
    spread_values: list[float | None] = []
    coverage_strict = []
    tolerances = gates["coverage_tolerance_primary_fields"]
    for field in B2_FIELDS:
        record = fields.get(field, {})
        comparator = deterministic_fields.get(field, {})
        ratio = _ratio(record.get("fair_crps"), comparator.get("mae"))
        fcrps_ratios.append(ratio)
        book.le(
            f"{prefix}.{field}.fCRPS_relative_to_paired_H2_MAE_relaxed",
            ratio,
            float(gates["fifth_field_fair_crps_relative_max"]),
            median_eligible=median_eligible,
        )
        spread = _finite(record.get("corrected_spread_skill", {}).get("ratio"))
        spread_values.append(spread)
        relaxed = gates["remaining_spread_skill_range"]
        book.between(
            f"{prefix}.{field}.spread_skill_relaxed",
            spread,
            float(relaxed[0]),
            float(relaxed[1]),
            median_eligible=median_eligible,
        )
        field_strict = True
        for interval, tolerance in tolerances.items():
            error = _coverage_error(record, interval)
            relaxed_pass = book.le(
                f"{prefix}.{field}.{interval}_coverage_error_relaxed",
                error,
                2.0 * float(tolerance),
                median_eligible=median_eligible,
            )
            field_strict = (
                field_strict
                and relaxed_pass
                and error is not None
                and error <= float(tolerance)
            )
        coverage_strict.append(field_strict)
    book.ge(
        f"{prefix}.fields_fCRPS_strictly_better_count",
        sum(value is not None and value < 1.0 for value in fcrps_ratios),
        float(gates["fields_required_fair_crps_better_than_paired_deterministic"]),
        median_eligible=median_eligible,
    )
    strict_spread = gates["primary_spread_skill_range"]
    book.ge(
        f"{prefix}.fields_primary_spread_skill_count",
        sum(
            value is not None
            and float(strict_spread[0]) <= value <= float(strict_spread[1])
            for value in spread_values
        ),
        float(gates["primary_fields_required_calibrated"]),
        median_eligible=median_eligible,
    )
    book.ge(
        f"{prefix}.fields_primary_coverage_count",
        sum(coverage_strict),
        float(gates["primary_fields_required_calibrated"]),
        median_eligible=median_eligible,
    )
    return book


def evaluate_field_family(
    score: Mapping[str, Any],
    comparator: Mapping[str, Any],
    best_uncompressed: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    overall_scope = score["regions"]["eligible_union"]
    overall = _field_scope_checks(
        overall_scope,
        comparator["overall"],
        best_uncompressed["overall"],
        gates,
        prefix="field.overall",
        median_eligible=True,
    )
    region_range = gates["region_I31_coverage_range"]
    for region in B2_PRIMARY_REGIONS:
        for field in B2_FIELDS:
            coverage = score["regions"][region]["fields"][field][
                "order_statistic_intervals"
            ]["I31"]["empirical_coverage"]
            overall.between(
                f"field.overall.{region}.{field}.I31_coverage",
                coverage,
                float(region_range[0]),
                float(region_range[1]),
            )
    prefixes = score["member_prefix_sensitivity_eligible_union"]
    for label in ("aggregate", *B2_FIELDS):
        m16_record = (
            prefixes["M16"]["aggregate"]
            if label == "aggregate"
            else prefixes["M16"]["fields"][label]
        )
        m32_record = (
            prefixes["M32"]["aggregate"]
            if label == "aggregate"
            else prefixes["M32"]["fields"][label]
        )
        metric = "equal_channel_fair_crps" if label == "aggregate" else "fair_crps"
        m16 = m16_record[metric]
        m32 = m32_record[metric]
        tolerance = 0.1 * abs(float(m32)) + 1.0e-8
        overall.le(
            f"field.overall.{label}.fCRPS_M16_M32_stability_fraction",
            abs(float(m16) - float(m32)) / tolerance,
            1.0,
        )

    blocks = []
    for index, (probabilistic, deterministic, uncompressed) in enumerate(
        zip(
            score["chronological_blocks_eligible_union"],
            comparator["chronological_blocks"],
            best_uncompressed["chronological_blocks"],
        )
    ):
        block = _field_scope_checks(
            probabilistic,
            deterministic,
            uncompressed,
            gates,
            prefix=f"field.block{index}",
            median_eligible=False,
        )
        blocks.append(block.record())
    block_count = sum(item["passes"] for item in blocks)
    overall.ge("field.blocks_passing", block_count, 5.0)
    record = overall.record()
    record["chronological_blocks"] = blocks
    record["blocks_passing"] = block_count
    record["blocks_required"] = 5
    return record


def _calibration_checks(
    book: CheckBook,
    calibration: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    prefix: str,
    median_eligible: bool,
) -> None:
    spread = (
        calibration.get("primary_M32", {})
        .get("corrected_spread_skill", {})
        .get("spread_skill_ratio")
    )
    spread_range = gates["material_calibration_spread_skill_range"]
    coverage_range = gates["material_calibration_I31_range"]
    coverage = (
        calibration.get("order_statistic_intervals", {})
        .get("I31", {})
        .get("empirical_coverage")
    )
    book.between(
        f"{prefix}.spread_skill",
        spread,
        float(spread_range[0]),
        float(spread_range[1]),
        median_eligible=median_eligible,
    )
    book.between(
        f"{prefix}.I31_coverage",
        coverage,
        float(coverage_range[0]),
        float(coverage_range[1]),
        median_eligible=median_eligible,
    )


def _spectral_scope_checks(
    spectral: Mapping[str, Any],
    materiality: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    prefix: str,
    median_eligible: bool,
    include_mc: bool,
) -> CheckBook:
    book = CheckBook()
    power_range = gates["member_expected_power_ratio_range"]
    for field in B2_FIELDS:
        for label, _, _ in B2_MODE_BANDS:
            if not materiality["fields"][field]["bands"][label]["material"]:
                continue
            band = spectral["toroidal_field_power"][field]["bands"][label]
            item_prefix = f"{prefix}.field.{field}.{label}"
            book.between(
                f"{item_prefix}.power_ratio",
                band.get("member_expected_power_ratio"),
                float(power_range[0]),
                float(power_range[1]),
                median_eligible=median_eligible,
            )
            book.ge(
                f"{item_prefix}.realization_coherence",
                band.get("ensemble_mean_realization_coherence_with_truth"),
                float(gates["ensemble_mean_realization_coherence_min"]),
                median_eligible=median_eligible,
            )
            calibration = band["per_target_band_power_calibration"]
            _calibration_checks(
                book,
                calibration,
                gates,
                prefix=f"{item_prefix}.calibration",
                median_eligible=median_eligible,
            )
            if include_mc:
                book.le(
                    f"{item_prefix}.fCRPS_M16_M32_stability_fraction",
                    _mc_fraction(calibration["M16_vs_M32_stability"]["fair_crps"]),
                    1.0,
                    median_eligible=median_eligible,
                )
    for first, second in B2_CROSS_PAIRS:
        pair = f"{first}-{second}"
        for label, _, _ in B2_MODE_BANDS:
            if not materiality["cross_fields"][pair]["bands"][label]["material"]:
                continue
            band = spectral["toroidal_cross_field"][pair]["bands"][label]
            item_prefix = f"{prefix}.cross.{pair}.{label}"
            book.le(
                f"{item_prefix}.absolute_phase_error_degrees",
                band.get("truth_amplitude_weighted_absolute_phase_error_degrees"),
                float(gates["cross_phase_error_degrees_max"]),
                median_eligible=median_eligible,
            )
            book.le(
                f"{item_prefix}.absolute_coherence_change",
                band.get("truth_amplitude_weighted_absolute_coherence_change"),
                float(gates["cross_coherence_absolute_change_max"]),
                median_eligible=median_eligible,
            )
            projections = band["per_target_cross_projection_calibration"]
            for component in ("real", "imaginary"):
                calibration = projections[component]
                component_prefix = f"{item_prefix}.{component}_calibration"
                _calibration_checks(
                    book,
                    calibration,
                    gates,
                    prefix=component_prefix,
                    median_eligible=median_eligible,
                )
                if include_mc:
                    book.le(
                        f"{component_prefix}.fCRPS_M16_M32_stability_fraction",
                        _mc_fraction(calibration["M16_vs_M32_stability"]["fair_crps"]),
                        1.0,
                        median_eligible=median_eligible,
                    )
    return book


def evaluate_spectral_family(
    score: Mapping[str, Any],
    materiality: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    overall = _spectral_scope_checks(
        score["overall"],
        materiality,
        gates,
        prefix="spectral.overall",
        median_eligible=True,
        include_mc=True,
    )
    blocks = []
    for index, block_score in enumerate(score["chronological_blocks"]):
        block = _spectral_scope_checks(
            block_score,
            materiality,
            gates,
            prefix=f"spectral.block{index}",
            median_eligible=False,
            include_mc=False,
        )
        blocks.append(block.record())
    block_count = sum(item["passes"] for item in blocks)
    overall.ge("spectral.blocks_passing", block_count, 5.0)
    record = overall.record()
    record["chronological_blocks"] = blocks
    record["blocks_passing"] = block_count
    record["blocks_required"] = 5
    return record


def _transport_scope_checks(
    transport: Mapping[str, Any],
    comparator: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    prefix: str,
    median_eligible: bool,
    include_mc: bool,
) -> CheckBook:
    book = CheckBook()
    strict_gates = gates["strict_faces"]
    sep_gates = gates["separatrix"]
    calibration_gates = gates["separatrix_calibration"]
    fcrps_ratios: list[float | None] = []
    calibrated: list[bool] = []
    for quantity in TRANSPORT_QUANTITIES:
        record = transport["quantities"][quantity]
        reductions = record["reductions"]
        strict = reductions["strict_face_contributions"]
        separatrix = reductions["separatrix_wedge"]
        strict_metrics = strict["ensemble_expected_paired_metrics"]
        sep_metrics = separatrix["ensemble_expected_paired_metrics"]
        quantity_prefix = f"{prefix}.{quantity}"
        book.le(
            f"{quantity_prefix}.strict_relative_l2",
            strict_metrics.get("relative_l2"),
            float(strict_gates["relative_l2_max"]),
            median_eligible=median_eligible,
        )
        book.ge(
            f"{quantity_prefix}.strict_correlation",
            strict_metrics.get("pearson_correlation"),
            float(strict_gates["correlation_min"]),
            median_eligible=median_eligible,
        )
        book.le(
            f"{quantity_prefix}.strict_sign_disagreement",
            strict_metrics.get("weighted_sign_disagreement"),
            float(strict_gates["weighted_sign_disagreement_max"]),
            median_eligible=median_eligible,
        )
        book.le(
            f"{quantity_prefix}.separatrix_relative_l2",
            sep_metrics.get("relative_l2"),
            float(sep_gates["relative_l2_max"]),
            median_eligible=median_eligible,
        )
        book.le(
            f"{quantity_prefix}.separatrix_absolute_normalized_bias",
            abs(float(sep_metrics["normalized_bias"]))
            if _finite(sep_metrics.get("normalized_bias")) is not None
            else None,
            float(sep_gates["absolute_normalized_bias_max"]),
            median_eligible=median_eligible,
        )
        book.ge(
            f"{quantity_prefix}.separatrix_correlation",
            sep_metrics.get("pearson_correlation"),
            float(sep_gates["correlation_min"]),
            median_eligible=median_eligible,
        )
        book.le(
            f"{quantity_prefix}.separatrix_sign_disagreement",
            sep_metrics.get("weighted_sign_disagreement"),
            float(sep_gates["weighted_sign_disagreement_max"]),
            median_eligible=median_eligible,
        )
        probabilistic = separatrix["ensemble_probabilistic_metrics"]
        ratio = _ratio(
            probabilistic.get("fair_crps"),
            comparator["quantities"][quantity].get("separatrix_absolute_error"),
        )
        fcrps_ratios.append(ratio)
        book.le(
            f"{quantity_prefix}.separatrix_fCRPS_relative_to_paired_H2_AE_relaxed",
            ratio,
            float(sep_gates["fourth_fair_crps_relative_max"]),
            median_eligible=median_eligible,
        )
        spread = _finite(probabilistic.get("corrected_spread_skill", {}).get("ratio"))
        spread_range = calibration_gates["spread_skill_range"]
        coverage27 = _coverage_error(probabilistic, "I27")
        coverage31 = _coverage_error(probabilistic, "I31")
        calibrated.append(
            spread is not None
            and float(spread_range[0]) <= spread <= float(spread_range[1])
            and coverage27 is not None
            and coverage27 <= float(calibration_gates["I27_coverage_tolerance"])
            and coverage31 is not None
            and coverage31 <= float(calibration_gates["I31_coverage_tolerance"])
        )
        book.finite(f"{quantity_prefix}.separatrix_spread_skill_finite", spread)
        book.finite(
            f"{quantity_prefix}.separatrix_I27_coverage_error_finite", coverage27
        )
        book.finite(
            f"{quantity_prefix}.separatrix_I31_coverage_error_finite", coverage31
        )
        book.boolean(
            f"{quantity_prefix}.separatrix_noncollapsed",
            probabilistic.get("spread_integrity", {}).get("nonzero_spread"),
        )
        event = record["upper_decile_event_conditioned"]
        book.boolean(f"{quantity_prefix}.event_defined", event.get("defined"))
        book.le(
            f"{quantity_prefix}.event_magnitude_relative_error",
            event.get("magnitude_relative_error"),
            float(gates["event_conditioned_magnitude_relative_error_max"]),
            median_eligible=median_eligible,
        )
        book.le(
            f"{quantity_prefix}.event_sign_disagreement",
            event.get("truth_magnitude_weighted_sign_disagreement"),
            float(gates["event_conditioned_weighted_sign_disagreement_max"]),
            median_eligible=median_eligible,
        )
        if include_mc:
            book.le(
                f"{quantity_prefix}.separatrix_fCRPS_M16_M32_stability_fraction",
                _mc_fraction(separatrix["M16_vs_M32_stability"]["fair_crps"]),
                1.0,
                median_eligible=median_eligible,
            )
    book.ge(
        f"{prefix}.separatrix_fCRPS_strictly_better_count",
        sum(value is not None and value < 1.0 for value in fcrps_ratios),
        float(sep_gates["fair_crps_better_than_paired_deterministic_required"]),
        median_eligible=median_eligible,
    )
    book.ge(
        f"{prefix}.separatrix_calibrated_count",
        sum(calibrated),
        float(sep_gates["probabilistically_calibrated_required"]),
        median_eligible=median_eligible,
    )
    return book


def evaluate_transport_family(
    score: Mapping[str, Any],
    comparator: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    overall = _transport_scope_checks(
        score["overall"],
        comparator["overall"],
        gates,
        prefix="transport.overall",
        median_eligible=True,
        include_mc=True,
    )
    blocks = []
    for index, (block_score, block_comparator) in enumerate(
        zip(score["chronological_blocks"], comparator["chronological_blocks"])
    ):
        block = _transport_scope_checks(
            block_score,
            block_comparator,
            gates,
            prefix=f"transport.block{index}",
            median_eligible=False,
            include_mc=False,
        )
        blocks.append(block.record())
    block_count = sum(item["passes"] for item in blocks)
    overall.ge("transport.blocks_passing", block_count, 5.0)
    record = overall.record()
    record["chronological_blocks"] = blocks
    record["blocks_passing"] = block_count
    record["blocks_required"] = 5
    return record


def _integrity_checks(
    *,
    result: Mapping[str, Any],
    score: Mapping[str, Any],
    training_run: Mapping[str, Any],
    materiality: Mapping[str, Any],
) -> dict[str, Any]:
    book = CheckBook()
    book.boolean("integrity.training_complete", training_run.get("training_complete"))
    book.boolean(
        "integrity.training_acceptance_not_prejudged",
        training_run.get("scientific_acceptance_evaluated") is False,
    )
    book.boolean(
        "integrity.result_scope",
        result.get("scope") == "B2_LDM_H2_full_probabilistic_evaluation_85604",
    )
    book.boolean(
        "integrity.result_status",
        result.get("status") == "completed_pending_frozen_acceptance_gate",
    )
    for name, expected in (
        ("development_run", "85604"),
        ("held_out_85606_read", False),
        ("guard_frames_read", False),
        ("target_truth_used_during_forecast_generation", False),
        ("truth_opened_only_after_forecast_hash", True),
        ("target_frames", [498, 624]),
        ("target_count", 126),
        ("ensemble_members", 32),
        ("physics_derived_training_loss_used", False),
        ("probabilistic_scientific_gate_evaluated", False),
        ("O3_launch_allowed", False),
        ("assimilation_allowed", False),
        ("diagnostic_ranking_allowed", False),
    ):
        book.boolean(f"integrity.result.{name}", result.get(name) == expected)
    book.boolean(
        "integrity.score_scope",
        score.get("scope") == "B2_truth_separated_probabilistic_scoring_85604",
    )
    for name, expected in (
        ("bounded_non_scientific_smoke", False),
        ("development_run", "85604"),
        ("held_out_85606_read", False),
        ("guard_frames_read", False),
        ("target_truth_used_during_forecast_generation", False),
        ("truth_opened_only_after_forecast_was_closed_and_hash_verified", True),
        ("training_performed", False),
        ("physics_derived_training_loss_used", False),
        ("target_frames", [498, 624]),
        ("target_count", 126),
    ):
        book.boolean(f"integrity.score.{name}", score.get(name) == expected)
    field_score = score["field_and_marginal_calibration"]
    for region in ("eligible_union", *B2_PRIMARY_REGIONS):
        for field in B2_FIELDS:
            book.boolean(
                f"integrity.nonzero_spread.{region}.{field}",
                field_score["regions"][region]["fields"][field]["spread_integrity"][
                    "nonzero_spread"
                ],
            )
    material_field_count = sum(
        item["material"]
        for field in B2_FIELDS
        for item in materiality["fields"][field]["bands"].values()
    )
    material_cross_count = sum(
        item["material"]
        for pair in materiality["cross_fields"].values()
        for item in pair["bands"].values()
    )
    book.boolean("integrity.material_field_band_exists", material_field_count > 0)
    book.boolean("integrity.material_cross_band_exists", material_cross_count > 0)
    return book.record()


def evaluate_b2_seed_acceptance(
    *,
    result: Mapping[str, Any],
    score: Mapping[str, Any],
    training_run: Mapping[str, Any],
    comparator_run: Mapping[str, Any],
    best_uncompressed: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(result.get("seed", -1))
    if seed not in (1701, 1702, 1703):
        raise ValueError("B2 acceptance seed differs")
    if (
        int(score.get("model_seed", -1)) != seed
        or int(training_run.get("seed", -1)) != seed
        or int(comparator_run.get("seed", -1)) != seed
    ):
        raise ValueError("B2 acceptance paired seed provenance differs")
    materiality = score["transport_event_thresholds"]["spectral_materiality"]
    gates = manifest["gates"]
    integrity = _integrity_checks(
        result=result,
        score=score,
        training_run=training_run,
        materiality=materiality,
    )
    field = evaluate_field_family(
        score["field_and_marginal_calibration"],
        comparator_run["field"],
        best_uncompressed["field"],
        gates["field"],
    )
    spectral = evaluate_spectral_family(
        score["spectral_and_cross_field"],
        materiality,
        gates["spectral"],
    )
    transport = evaluate_transport_family(
        score["memberwise_transport"],
        comparator_run["transport"],
        gates["transport"],
    )
    families = {"field": field, "spectral": spectral, "transport": transport}
    numeric_checks = {
        item["name"]: item
        for family in families.values()
        for item in family["checks"]
        if item["kind"] == "numeric" and item["median_eligible"]
    }
    all_required_numeric = [
        item
        for family in families.values()
        for item in (
            list(family["checks"])
            + [
                check
                for block in family["chronological_blocks"]
                for check in block["checks"]
            ]
        )
        if item["kind"] == "numeric"
    ]
    all_numeric_finite = all(item["finite"] for item in all_required_numeric)
    field_by_name = {item["name"]: item for item in field["checks"]}
    field_catastrophic = all(
        field_by_name[name]["value"] is not None
        and field_by_name[name]["value"] <= 1.20
        for name in (
            "field.overall.aggregate_rmse_relative_to_paired_H2",
            "field.overall.aggregate_mae_relative_to_paired_H2",
        )
    )
    sep_relative = [
        item["value"]
        for item in transport["checks"]
        if item["name"].endswith(".separatrix_relative_l2")
        and item["name"].startswith("transport.overall.")
    ]
    transport_catastrophic = len(sep_relative) == len(TRANSPORT_QUANTITIES) and all(
        value is not None and value <= 0.60 for value in sep_relative
    )
    field_collapse_free = all(
        item["passes"]
        for item in integrity["checks"]
        if "nonzero_spread" in item["name"]
    )
    transport_collapse_free = all(
        item["passes"]
        for item in transport["checks"]
        if item["name"].endswith(".separatrix_noncollapsed")
    )
    collapse_free = field_collapse_free and transport_collapse_free
    passes = integrity["passes"] and all(
        family["passes"] for family in families.values()
    )
    return {
        "seed": seed,
        "passes_complete_per_seed_gate": passes,
        "integrity": integrity,
        "families": families,
        "numeric_checks_for_architecture_median": numeric_checks,
        "catastrophic_bounds": {
            "integrity_passes": integrity["passes"],
            "all_required_numeric_metrics_finite": all_numeric_finite,
            "ensemble_collapse_absent": collapse_free,
            "aggregate_field_RMSE_and_MAE_within_1p20_paired_H2": (field_catastrophic),
            "all_separatrix_relative_l2_at_most_0p60": transport_catastrophic,
            "passes": (
                integrity["passes"]
                and all_numeric_finite
                and collapse_free
                and field_catastrophic
                and transport_catastrophic
            ),
        },
    }


def _evaluate_from_template(
    values: Sequence[float], template: Mapping[str, Any]
) -> dict[str, Any]:
    finite_values = [_finite(value) for value in values]
    if len(values) != 3 or any(value is None for value in finite_values):
        return {
            **dict(template),
            "seed_values": finite_values,
            "value": None,
            "finite": False,
            "passes": False,
        }
    median = float(np.median(np.asarray(finite_values, dtype=np.float64)))
    operator = template["operator"]
    if operator == "<=":
        passed = median <= float(template["upper"])
    elif operator == "<":
        passed = median < float(template["upper"])
    elif operator == ">=":
        passed = median >= float(template["lower"])
    elif operator == "range":
        passed = float(template["lower"]) <= median <= float(template["upper"])
    else:
        raise ValueError("B2 median check operator differs")
    return {
        **dict(template),
        "seed_values": finite_values,
        "value": median,
        "finite": True,
        "passes": passed,
    }


def evaluate_b2_architecture_acceptance(
    seed_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = tuple(seed_records)
    if tuple(int(item.get("seed", -1)) for item in records) != (1701, 1702, 1703):
        raise ValueError("B2 architecture gate requires ordered seeds 1701..1703")
    pass_count = sum(bool(item["passes_complete_per_seed_gate"]) for item in records)
    check_names = tuple(records[0]["numeric_checks_for_architecture_median"])
    if any(
        tuple(item["numeric_checks_for_architecture_median"]) != check_names
        for item in records[1:]
    ):
        raise ValueError("B2 seed median-check schemas differ")
    median_checks = []
    for name in check_names:
        templates = [
            item["numeric_checks_for_architecture_median"][name] for item in records
        ]
        template = templates[0]
        if any(
            any(
                other.get(key) != template.get(key)
                for key in ("operator", "lower", "upper")
            )
            for other in templates[1:]
        ):
            raise ValueError(f"B2 median threshold differs for {name}")
        values = [item.get("value") for item in templates]
        median_checks.append(_evaluate_from_template(values, template))
    median_passes = bool(median_checks) and all(
        item["passes"] for item in median_checks
    )
    nonpassing = [item for item in records if not item["passes_complete_per_seed_gate"]]
    remaining_noncatastrophic = all(
        item["catastrophic_bounds"]["passes"] for item in nonpassing
    )
    architecture_passes = (
        pass_count >= 2 and median_passes and remaining_noncatastrophic
    )
    return {
        "schema_version": 1,
        "scope": "phase3_B2_LDM_H2_frozen_architecture_acceptance_85604",
        "seed_count": 3,
        "seeds": [1701, 1702, 1703],
        "complete_seed_gate_pass_count": pass_count,
        "complete_seed_gate_passes_required": 2,
        "median_numerical_gate": {
            "passes": median_passes,
            "check_count": len(median_checks),
            "failed_check_count": sum(not item["passes"] for item in median_checks),
            "checks": median_checks,
        },
        "nonpassing_seed_catastrophic_bounds_pass": remaining_noncatastrophic,
        "architecture_passes_one_step_B2_gate": architecture_passes,
        "short_O3_protocol_may_be_frozen": architecture_passes,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "held_out_85606_access_allowed": False,
        "per_seed": list(records),
    }
