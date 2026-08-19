"""Frozen H-det/H-prob reduction for the B4 PDE-Refiner.

This module does not read files, run inference, rescore forecasts, or fit
thresholds.  It adapts the prospectively frozen B4 threshold names to the
byte-locked B2 numerical reducers, then projects their explicit checks into
two independent hypotheses:

* H-det: mean/realization fidelity and stagewise physical repair;
* H-prob: ensemble calibration and Monte-Carlo stability.

The same five of six chronological blocks must pass all applicable field,
spectral, and transport checks for a hypothesis.  Passing different blocks in
different metric families is therefore insufficient.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from .b2_acceptance_gate import (
    CheckBook,
    evaluate_field_family,
    evaluate_spectral_family,
)
from .b2_acceptance_gate_event_eligibility import (
    EVENT_BLOCK_POLICY,
    evaluate_transport_family_event_eligible,
)
from .b2_field_metrics import B2_FIELDS, B2_PRIMARY_REGIONS


B4_EVALUATION_SCOPE = "B4_PDE_Refiner_H1_full_one_step_evaluation_85604"
B4_FINAL_SCORE_SCOPE = "B4_PDE_Refiner_H1_final_M32_scoring_85604"
B4_STAGE_SCORE_SCOPE = "B4_PDE_Refiner_H1_stagewise_H_det_repair_scoring_85604"
B4_GENERATION_SCOPE = "B4_PDE_Refiner_H1_one_step_forecast_generation_85604"
B4_TRAINING_SCOPE = "B4_PDE_Refiner_H1_seed1701_full_training_85604"
B4_COMPARATOR_SCOPE = "phase3_B3_frozen_matched_H1_comparators_85604"
B4_MANIFEST_SHA256 = (
    "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229"
)
B4_PROTOCOL_SHA256 = (
    "ffa56b2111074253a70c7453f1e36f91ca747ec59a68d632288764d60387aad1"
)
B4_EVENT_THRESHOLD_SHA256 = (
    "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb"
)
B4_SCIENTIFIC_SEED_SHA256 = (
    "a1871e069bce6244073bfe1aa835a53c1d7a59302b01f6a366b3dc88297b6205"
)


def adapt_b4_numerical_gates(
    gates: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Map frozen B4 H-det/H-prob names to unchanged B2 reducers."""

    if set(gates) != {
        "H_det",
        "H_prob",
        "blocks_required_passing",
        "blocks_total",
        "integrity",
    }:
        raise ValueError("B4 numerical gate families differ")
    if gates["blocks_required_passing"] != 5 or gates["blocks_total"] != 6:
        raise ValueError("B4 chronological gate differs")
    det = gates["H_det"]
    prob = gates["H_prob"]
    return {
        "field": {
            "aggregate_mean_rmse_relative_to_paired_deterministic_max": det[
                "aggregate_mean_RMSE_relative_to_parent_H1_max"
            ],
            "aggregate_mean_mae_relative_to_paired_deterministic_max": det[
                "aggregate_mean_MAE_relative_to_parent_H1_max"
            ],
            "fields_required_fair_crps_better_than_paired_deterministic": prob[
                "fields_required_fair_CRPS_better_than_parent_H1"
            ],
            "fifth_field_fair_crps_relative_max": prob[
                "fifth_field_fair_CRPS_relative_max"
            ],
            "primary_spread_skill_range": prob[
                "primary_field_spread_skill_range"
            ],
            "remaining_spread_skill_range": prob[
                "remaining_field_spread_skill_range"
            ],
            "primary_fields_required_calibrated": prob[
                "primary_fields_required_calibrated"
            ],
            "coverage_tolerance_primary_fields": prob[
                "primary_field_coverage_tolerance"
            ],
            "region_I31_coverage_range": prob["primary_region_I31_range"],
        },
        "spectral": {
            "member_expected_power_ratio_range": det[
                "material_power_ratio_range"
            ],
            "ensemble_mean_realization_coherence_min": det[
                "material_realization_coherence_min"
            ],
            "cross_phase_error_degrees_max": det[
                "material_cross_phase_error_degrees_max"
            ],
            "cross_coherence_absolute_change_max": det[
                "material_cross_coherence_absolute_change_max"
            ],
            "material_calibration_spread_skill_range": prob[
                "material_power_and_cross_projection_spread_skill_range"
            ],
            "material_calibration_I31_range": prob[
                "material_power_and_cross_projection_I31_range"
            ],
        },
        "transport": {
            "strict_faces": deepcopy(det["strict_faces"]),
            "separatrix": {
                **deepcopy(det["separatrix"]),
                "fourth_fair_crps_relative_max": prob[
                    "fourth_separatrix_fair_CRPS_relative_max"
                ],
                "fair_crps_better_than_paired_deterministic_required": prob[
                    "separatrix_fair_CRPS_better_than_parent_H1_required"
                ],
            },
            "separatrix_calibration": {
                "spread_skill_range": prob["separatrix_spread_skill_range"],
                "I27_coverage_tolerance": prob[
                    "separatrix_I27_coverage_tolerance"
                ],
                "I31_coverage_tolerance": prob[
                    "separatrix_I31_coverage_tolerance"
                ],
                "probabilistically_calibrated_required": prob[
                    "separatrix_calibrated_required"
                ],
            },
            "event_conditioned_magnitude_relative_error_max": det[
                "event_magnitude_relative_error_max"
            ],
            "event_conditioned_weighted_sign_disagreement_max": det[
                "event_weighted_sign_disagreement_max"
            ],
        },
    }


def _without_block_summary(name: str) -> bool:
    return not name.endswith(".blocks_passing")


def _field_det(name: str) -> bool:
    return _without_block_summary(name) and (
        ".aggregate_rmse_relative_to_" in name
        or ".aggregate_mae_relative_to_" in name
    )


def _field_prob(name: str) -> bool:
    return _without_block_summary(name) and not _field_det(name)


def _spectral_det(name: str) -> bool:
    return _without_block_summary(name) and any(
        token in name
        for token in (
            ".power_ratio",
            ".realization_coherence",
            ".absolute_phase_error_degrees",
            ".absolute_coherence_change",
        )
    )


def _spectral_prob(name: str) -> bool:
    return _without_block_summary(name) and not _spectral_det(name)


def _transport_det(name: str) -> bool:
    return _without_block_summary(name) and any(
        token in name
        for token in (
            ".strict_relative_l2",
            ".strict_correlation",
            ".strict_sign_disagreement",
            ".separatrix_relative_l2",
            ".separatrix_absolute_normalized_bias",
            ".separatrix_correlation",
            ".separatrix_sign_disagreement",
            ".event_",
            ".zero_event_",
        )
    )


def _transport_prob(name: str) -> bool:
    return _without_block_summary(name) and any(
        token in name
        for token in (
            ".separatrix_fCRPS_",
            ".separatrix_spread_skill_finite",
            ".separatrix_I27_coverage_error_finite",
            ".separatrix_I31_coverage_error_finite",
            ".separatrix_noncollapsed",
            ".separatrix_calibrated_count",
        )
    )


def _check_record(checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "passes": bool(checks) and all(item.get("passes") is True for item in checks),
        "check_count": len(checks),
        "failed_check_count": sum(item.get("passes") is not True for item in checks),
        "checks": checks,
    }


def _project_family(
    record: Mapping[str, Any],
    predicate: Callable[[str], bool],
) -> dict[str, Any]:
    """Project a combined reducer record without changing check values."""

    overall_checks = [
        deepcopy(item)
        for item in record.get("checks", [])
        if predicate(str(item.get("name", "")))
    ]
    blocks = []
    for block in record.get("chronological_blocks", []):
        checks = [
            deepcopy(item)
            for item in block.get("checks", [])
            if predicate(str(item.get("name", "")))
        ]
        blocks.append(_check_record(checks))
    if len(blocks) != 6:
        raise ValueError("B4 projected family requires six chronological blocks")
    projected = _check_record(overall_checks)
    projected["chronological_blocks"] = blocks
    projected["blocks_passing"] = sum(item["passes"] for item in blocks)
    projected["blocks_required"] = 5
    projected["passes_overall"] = projected["passes"]
    projected["passes_temporally"] = projected["blocks_passing"] >= 5
    projected["passes"] = bool(
        projected["passes_overall"] and projected["passes_temporally"]
    )
    return projected


def evaluate_b4_numerical_families(
    *,
    score: Mapping[str, Any],
    comparator: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Run locked reducers once and project their checks into H-det/H-prob."""

    numerical = adapt_b4_numerical_gates(manifest.get("gates", {}))
    materiality = score["transport_event_thresholds"]["spectral_materiality"]
    combined = {
        "field": evaluate_field_family(
            score["field_and_marginal_calibration"],
            comparator["field"],
            comparator["best_uncompressed"]["field"],
            numerical["field"],
        ),
        "spectral": evaluate_spectral_family(
            score["spectral_and_cross_field"],
            materiality,
            numerical["spectral"],
        ),
        "transport": evaluate_transport_family_event_eligible(
            score["memberwise_transport"],
            comparator["transport"],
            numerical["transport"],
        ),
    }
    return {
        "H_det": {
            "field": _project_family(combined["field"], _field_det),
            "spectral": _project_family(combined["spectral"], _spectral_det),
            "transport": _project_family(combined["transport"], _transport_det),
        },
        "H_prob": {
            "field": _project_family(combined["field"], _field_prob),
            "spectral": _project_family(combined["spectral"], _spectral_prob),
            "transport": _project_family(combined["transport"], _transport_prob),
        },
    }


def _material_band_counts(materiality: Mapping[str, Any]) -> tuple[int, int]:
    field_count = sum(
        item.get("material") is True
        for field in B2_FIELDS
        for item in materiality.get("fields", {}).get(field, {}).get("bands", {}).values()
    )
    cross_count = sum(
        item.get("material") is True
        for pair in materiality.get("cross_fields", {}).values()
        for item in pair.get("bands", {}).values()
    )
    return field_count, cross_count


def _wandb_finished(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("required") is True
        and record.get("mode") == "online"
        and record.get("remote_presence_verified_after_finish") is True
        and record.get("remote_state_after_finish") == "finished"
        and record.get("local_artifacts_are_scientific_authority") is True
    )


def evaluate_b4_integrity(
    *,
    result: Mapping[str, Any],
    score: Mapping[str, Any],
    stage_score: Mapping[str, Any],
    training: Mapping[str, Any],
    generation: Mapping[str, Any],
    comparator: Mapping[str, Any],
    manifest: Mapping[str, Any],
    training_wandb: Mapping[str, Any],
    evaluation_wandb: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate B4 provenance and truth-separation prerequisites."""

    book = CheckBook()
    book.boolean(
        "integrity.manifest.status",
        manifest.get("protocol_status")
        == "frozen_after_passing_B4_smoke_before_full_training_checkpoint_selection_or_scientific_evaluation_implementation",
    )
    for name, expected in (
        ("development_run", "85604"),
        ("sequestered_run", "85606"),
        ("held_out_85606_access_allowed", False),
        ("full_training_authorized", True),
        ("scientific_one_step_evaluation_authorized", True),
    ):
        book.boolean(f"integrity.manifest.{name}", manifest.get(name) == expected)
    data = manifest.get("data", {})
    book.boolean(
        "integrity.manifest.fields",
        data.get("fields") == ["Ne", "Pe", "Pi", "phi", "Vi"],
    )
    book.boolean("integrity.manifest.no_time_input", data.get("absolute_time_input_allowed") is False)
    book.boolean("integrity.manifest.zperiod_5", data.get("zperiod") == 5)
    book.boolean("integrity.manifest.n_eq_5k", data.get("mode_mapping") == "n=5k")

    book.boolean("integrity.training.scope", training.get("scope") == B4_TRAINING_SCOPE)
    for name, expected in (
        ("completed_epochs", 100),
        ("completed_optimizer_steps", 2700),
        ("checkpoint_reload_bitwise_exact", True),
        ("codec_bitwise_unchanged", True),
        ("all_four_training_levels_exercised", True),
        ("parent_parameter_gradient_seen", True),
        ("refinement_parameter_gradient_seen", True),
        ("physics_derived_loss_used", False),
        ("target_truth_used_as_model_input", False),
        ("absolute_time_used_as_model_input", False),
        ("held_out_85606_read", False),
        ("scientific_result", False),
        ("H_det_evaluated", False),
        ("H_prob_evaluated", False),
    ):
        book.boolean(f"integrity.training.{name}", training.get(name) == expected)
    book.boolean(
        "integrity.training.objective",
        training.get("config", {}).get("training_loss")
        == "uniform_level_explicit_standardized_latent_MSE",
    )
    book.boolean(
        "integrity.training.selected_checkpoint_final_candidate",
        training.get("selected_completed_epoch") == 100
        and training.get("selected_epoch") == 99,
    )
    book.boolean(
        "integrity.training.scientific_bank_not_selection_bank",
        training.get("validation_seed_bank", {}).get("sha256")
        != B4_SCIENTIFIC_SEED_SHA256,
    )

    book.boolean("integrity.evaluation.scope", result.get("scope") == B4_EVALUATION_SCOPE)
    for name, expected in (
        ("status", "completed_pending_frozen_H_det_H_prob_reduction"),
        ("scientific_authority", True),
        ("bounded_non_scientific_smoke", False),
        ("development_run", "85604"),
        ("held_out_85606_read", False),
        ("guard_frames_read", False),
        ("target_truth_used_during_forecast_generation", False),
        ("truth_opened_only_after_both_forecast_hashes", True),
        ("absolute_time_used_as_model_input", False),
        ("target_frames", [498, 624]),
        ("target_count", 126),
        ("final_ensemble_members", 32),
        ("stage_prefix_members", 4),
        ("member_prefixes_regenerated", False),
        ("posthoc_calibration_applied", False),
        ("physics_derived_training_loss_used", False),
        ("full_evaluation_preconditions_passed", True),
        ("H_det_evaluated", False),
        ("H_prob_evaluated", False),
        ("O3_launch_allowed", False),
        ("assimilation_allowed", False),
        ("diagnostic_ranking_allowed", False),
        ("seed", 1701),
    ):
        book.boolean(f"integrity.evaluation.{name}", result.get(name) == expected)
    book.boolean(
        "integrity.evaluation.checkpoint_matches_training",
        result.get("selected_checkpoint", {}).get("sha256")
        == training.get("selected_checkpoint", {}).get("sha256"),
    )
    history = result.get("training_history_audit", {})
    book.boolean(
        "integrity.evaluation.complete_history",
        history.get("epochs") == 100
        and history.get("optimizer_steps") == 2700
        and history.get("earliest_validation_minimum_epoch")
        == training.get("selected_epoch")
        and history.get("finite") is True,
    )
    book.boolean(
        "integrity.evaluation.manifest_lock",
        result.get("evaluation_manifest", {}).get("sha256") == B4_MANIFEST_SHA256,
    )
    book.boolean(
        "integrity.evaluation.protocol_lock",
        result.get("evaluation_protocol", {}).get("sha256") == B4_PROTOCOL_SHA256,
    )
    book.boolean(
        "integrity.evaluation.event_threshold_lock",
        result.get("event_threshold_result", {}).get("sha256")
        == B4_EVENT_THRESHOLD_SHA256,
    )
    book.boolean(
        "integrity.evaluation.scientific_seed_bank",
        result.get("scientific_seed_bank", {}).get("seed") == 41032
        and result.get("scientific_seed_bank", {}).get("shape") == [126, 32, 3]
        and result.get("scientific_seed_bank", {}).get("sha256")
        == B4_SCIENTIFIC_SEED_SHA256,
    )

    book.boolean("integrity.generation.scope", generation.get("scope") == B4_GENERATION_SCOPE)
    for name, expected in (
        ("bounded_non_scientific_smoke", False),
        ("development_run", "85604"),
        ("held_out_85606_read", False),
        ("guard_frames_read", False),
        ("target_truth_used_as_model_input", False),
        ("absolute_time_used_as_model_input", False),
        ("member_interaction", False),
        ("member_prefixes_regenerated", False),
        ("posthoc_calibration", False),
        ("target_frames", [498, 624]),
        ("target_count", 126),
    ):
        book.boolean(f"integrity.generation.{name}", generation.get(name) == expected)
    book.boolean(
        "integrity.generation.final_shape",
        generation.get("final_forecast", {}).get("shape")
        == [126, 32, 1, 5, 64, 32, 88],
    )
    book.boolean(
        "integrity.generation.stage_shape",
        generation.get("stage_forecast", {}).get("shape")
        == [126, 4, 4, 5, 64, 32, 88],
    )
    book.boolean(
        "integrity.generation.prefix_and_shared_level0",
        generation.get("stage_forecast", {}).get("M4_stage3_bitwise_prefix_of_M32")
        is True
        and generation.get("stage_forecast", {}).get("level0_bitwise_shared_across_members")
        is True,
    )
    book.boolean(
        "integrity.generation.scientific_seed",
        generation.get("scientific_seed_bank", {}).get("sha256")
        == B4_SCIENTIFIC_SEED_SHA256
        and generation.get("scientific_seed_bank", {}).get("complete_M32_generated_once")
        is True
        and generation.get("scientific_seed_bank", {}).get("independent_of_checkpoint_selection_noise")
        is True,
    )
    book.boolean(
        "integrity.generation.inference_contract",
        generation.get("inference", {}).get("kind")
        == "three_stage_explicit_latent_PDE_Refiner"
        and generation.get("inference", {}).get(
            "unamortized_member_equivalent_transition_evaluations"
        )
        == 128
        and generation.get("inference", {}).get(
            "shared_level0_member_equivalent_transition_evaluations"
        )
        == 97,
    )

    book.boolean("integrity.final_score.scope", score.get("scope") == B4_FINAL_SCORE_SCOPE)
    book.boolean("integrity.stage_score.scope", stage_score.get("scope") == B4_STAGE_SCORE_SCOPE)
    for label, record in (("final_score", score), ("stage_score", stage_score)):
        for name, expected in (
            ("bounded_non_scientific_smoke", False),
            ("development_run", "85604"),
            ("held_out_85606_read", False),
            ("guard_frames_read", False),
            ("target_truth_used_during_forecast_generation", False),
            ("training_performed", False),
            ("physics_derived_training_loss_used", False),
            ("model_seed", 1701),
            ("target_frames", [498, 624]),
            ("target_count", 126),
        ):
            book.boolean(f"integrity.{label}.{name}", record.get(name) == expected)
    book.boolean(
        "integrity.final_score.truth_separation",
        score.get("truth_opened_only_after_forecast_was_closed_and_hash_verified")
        is True,
    )
    book.boolean(
        "integrity.stage_score.truth_separation",
        stage_score.get("truth_opened_only_after_both_forecasts_closed_and_hash_verified")
        is True,
    )
    book.boolean(
        "integrity.final_score.forecast_hash",
        score.get("forecast_artifact", {}).get("sha256")
        == result.get("final_forecast", {}).get("sha256"),
    )
    book.boolean(
        "integrity.stage_score.forecast_hash",
        stage_score.get("stage_artifact", {}).get("sha256")
        == result.get("stage_forecast", {}).get("sha256"),
    )
    expected_sources = manifest.get("metric_engine", {}).get("source_sha256")
    book.boolean(
        "integrity.final_score.metric_engine",
        score.get("metric_engine", {}).get("numerical_definitions_changed_for_B4_final")
        is False
        and score.get("metric_engine", {}).get("source_sha256") == expected_sources
        and result.get("metric_source_sha256") == expected_sources,
    )
    book.boolean(
        "integrity.stage_score.metric_engine",
        stage_score.get("metric_sources") == expected_sources,
    )
    book.boolean(
        "integrity.stage_score.gate_evaluated",
        stage_score.get("stagewise_repair", {}).get("gate_evaluated") is True
        and isinstance(stage_score.get("stagewise_repair", {}).get("passes"), bool),
    )

    parent = manifest.get("comparators", {}).get("primary_deterministic_parent", {})
    book.boolean("integrity.comparator.scope", comparator.get("scope") == B4_COMPARATOR_SCOPE)
    book.boolean(
        "integrity.comparator.frozen",
        comparator.get("status") == "completed_before_B3_scientific_acceptance"
        and comparator.get("development_run") == "85604"
        and comparator.get("held_out_85606_read") is False
        and comparator.get("B3_forecasts_or_scores_read") is False
        and comparator.get("deterministic_model_retrained") is False
        and comparator.get("deterministic_checkpoint_reselected") is False
        and comparator.get("uncompressed_reference_reselected") is False
        and comparator.get("scientific_acceptance_evaluated") is False,
    )
    book.boolean(
        "integrity.comparator.parent_lock",
        comparator.get("forecast", {}).get("sha256") == parent.get("forecast_sha256")
        and comparator.get("score", {}).get("sha256") == parent.get("score_sha256")
        and comparator.get("arm") == "C5P-H1"
        and comparator.get("seed") == 1701,
    )
    book.boolean(
        "integrity.comparator.uncompressed_lock",
        comparator.get("best_uncompressed", {}).get("name")
        == "training_only_toroidal_spectral_AR1",
    )

    block_counts = {
        "field_score": len(
            score.get("field_and_marginal_calibration", {}).get(
                "chronological_blocks_eligible_union", []
            )
        ),
        "spectral_score": len(
            score.get("spectral_and_cross_field", {}).get("chronological_blocks", [])
        ),
        "transport_score": len(
            score.get("memberwise_transport", {}).get("chronological_blocks", [])
        ),
        "parent_field": len(comparator.get("field", {}).get("chronological_blocks", [])),
        "parent_transport": len(
            comparator.get("transport", {}).get("chronological_blocks", [])
        ),
        "uncompressed_field": len(
            comparator.get("best_uncompressed", {})
            .get("field", {})
            .get("chronological_blocks", [])
        ),
    }
    for name, count in block_counts.items():
        book.boolean(f"integrity.six_blocks.{name}", count == 6)

    field_score = score.get("field_and_marginal_calibration", {})
    for region in ("eligible_union", *B2_PRIMARY_REGIONS):
        for field in B2_FIELDS:
            book.boolean(
                f"integrity.nonzero_spread.{region}.{field}",
                field_score.get("regions", {})
                .get(region, {})
                .get("fields", {})
                .get(field, {})
                .get("spread_integrity", {})
                .get("nonzero_spread")
                is True,
            )
    materiality = score.get("transport_event_thresholds", {}).get(
        "spectral_materiality", {}
    )
    field_bands, cross_bands = _material_band_counts(materiality)
    book.boolean("integrity.material_field_band_exists", field_bands > 0)
    book.boolean("integrity.material_cross_band_exists", cross_bands > 0)
    book.boolean(
        "integrity.wandb.training",
        _wandb_finished(training_wandb) and training_wandb.get("epochs_logged") == 100,
    )
    book.boolean(
        "integrity.wandb.evaluation",
        _wandb_finished(evaluation_wandb)
        and evaluation_wandb.get("evaluation_mode") == "full",
    )
    record = book.record()
    record["material_field_band_count"] = field_bands
    record["material_cross_band_count"] = cross_bands
    record["chronological_block_counts"] = block_counts
    return record


def _hypothesis_record(
    *,
    name: str,
    integrity: Mapping[str, Any],
    families: Mapping[str, Mapping[str, Any]],
    stage_score: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blocks = []
    for index in range(6):
        family_passes = {
            family: bool(record["chronological_blocks"][index]["passes"])
            for family, record in families.items()
        }
        blocks.append(
            {
                "index": index,
                "target_frames": [498 + 21 * index, 519 + 21 * index],
                "family_passes": family_passes,
                "passes": all(family_passes.values()),
            }
        )
    joint_count = sum(item["passes"] for item in blocks)
    overall_family_passes = {
        family: bool(record["passes_overall"])
        for family, record in families.items()
    }
    checks = {
        "integrity": integrity.get("passes") is True,
        "all_families_pass_overall": all(overall_family_passes.values()),
        "joint_blocks_passing_at_least_five": joint_count >= 5,
    }
    if stage_score is not None:
        checks["stagewise_repair"] = (
            stage_score.get("stagewise_repair", {}).get("gate_evaluated") is True
            and stage_score.get("stagewise_repair", {}).get("passes") is True
        )
    return {
        "hypothesis": name,
        "passes": all(checks.values()),
        "checks": checks,
        "overall_family_passes": overall_family_passes,
        "chronological_blocks": blocks,
        "joint_blocks_passing": joint_count,
        "joint_blocks_required": 5,
        "families": deepcopy(dict(families)),
    }


def evaluate_b4_one_seed_acceptance(
    *,
    result: Mapping[str, Any],
    score: Mapping[str, Any],
    stage_score: Mapping[str, Any],
    training: Mapping[str, Any],
    generation: Mapping[str, Any],
    comparator: Mapping[str, Any],
    manifest: Mapping[str, Any],
    training_wandb: Mapping[str, Any],
    evaluation_wandb: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply both frozen B4 hypotheses without rescoring or data access."""

    if result.get("seed") != 1701 or score.get("model_seed") != 1701:
        raise ValueError("B4 one-seed gate requires seed 1701")
    integrity = evaluate_b4_integrity(
        result=result,
        score=score,
        stage_score=stage_score,
        training=training,
        generation=generation,
        comparator=comparator,
        manifest=manifest,
        training_wandb=training_wandb,
        evaluation_wandb=evaluation_wandb,
    )
    families = evaluate_b4_numerical_families(
        score=score,
        comparator=comparator,
        manifest=manifest,
    )
    h_det = _hypothesis_record(
        name="H_det",
        integrity=integrity,
        families=families["H_det"],
        stage_score=stage_score,
    )
    h_prob = _hypothesis_record(
        name="H_prob",
        integrity=integrity,
        families=families["H_prob"],
        stage_score=None,
    )
    joint = bool(h_det["passes"] and h_prob["passes"])
    if joint:
        disposition = manifest["decision_rule"]["joint_pass"]
    elif h_det["passes"]:
        disposition = manifest["decision_rule"]["H_det_pass_H_prob_fail"]
    elif h_prob["passes"]:
        disposition = manifest["decision_rule"]["H_prob_pass_H_det_fail"]
    else:
        disposition = manifest["decision_rule"]["H_det_fail"]
    return {
        "schema_version": 1,
        "scope": "phase3_B4_PDE_Refiner_H1_seed1701_frozen_H_det_H_prob_gate_85604",
        "seed": 1701,
        "event_block_policy": EVENT_BLOCK_POLICY,
        "numerical_gate_engine": {
            "identity": "byte_locked_B2_reducers_with_B4_threshold_adapter_and_independent_projection",
            "numerical_reducers_changed": False,
            "threshold_values_changed_during_adaptation": False,
            "same_joint_blocks_required_across_families": True,
        },
        "integrity": integrity,
        "H_det": h_det,
        "H_prob": h_prob,
        "joint_H_det_H_prob_pass": joint,
        "disposition": disposition,
        "seed1702_1703_replication_protocol_may_be_written": joint,
        "seed1702_1703_training_authorized": False,
        "O3_protocol_may_be_written": bool(
            h_det["passes"] and not h_prob["passes"]
        ),
        "O3_launch_allowed": False,
        "held_out_85606_access_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
