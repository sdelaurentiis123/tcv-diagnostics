"""Frozen one-seed acceptance reduction for the B5 residual EDM."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

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
from .fgn_acceptance_gate import adapt_b3_numerical_gates


B5_EVALUATION_SCOPE = "B5_residual_EDM_full_one_step_evaluation_85604"
B5_SCORE_SCOPE = "B5_residual_EDM_truth_separated_probabilistic_scoring_85604"
B5_GENERATION_SCOPE = "B5_residual_EDM_one_step_M32_forecast_generation_85604"
B5_TRAINING_SCOPE = "B5_seed1701_full_training_and_data_only_selection_85604"
B5_COMPARATOR_SCOPE = "phase3_B3_frozen_matched_H1_comparators_85604"
B5_MANIFEST_SHA256 = "61f1fa565e2bcff008cbe72909daa97362dabe96d160a9beee4a3d5aa87d1334"
B5_PROTOCOL_SHA256 = "faab336bf3ae1a49008eff0e6604d48d9c475aa83732184668c4c2e444c928b9"
B5_EVENT_THRESHOLD_SHA256 = (
    "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb"
)
B5_SCIENTIFIC_SEED_SHA256 = (
    "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
)
B5_COMPARATOR_SHA256 = (
    "2b04c10971e6d38ee439e33aa0b5331305acf16b38a96e7952fb26046049b5d2"
)


def validate_inherited_b3_thresholds(
    *,
    b5_manifest: Mapping[str, Any],
    inherited_b3_gates: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Prove the adapted complete B3 gate matches the B5 frozen summary."""

    acceptance = b5_manifest.get("acceptance", {})
    if (
        acceptance.get("threshold_identity")
        != "same_numerical_field_joint_physics_transport_and_Monte_Carlo_thresholds_as_frozen_B3_gate"
        or acceptance.get("source_protocol_sha256")
        != "db717c5605ad9653d2b051ec13254b43bf230f514cb173d295e95d3c68af8030"
        or acceptance.get("temporal_blocks_required") != 5
        or acceptance.get("temporal_blocks_total") != 6
    ):
        raise ValueError("B5 inherited-threshold identity differs")
    numerical = adapt_b3_numerical_gates(inherited_b3_gates)
    field = numerical["field"]
    spectral = numerical["spectral"]
    transport = numerical["transport"]
    comparisons = (
        (
            field["aggregate_mean_mae_relative_to_paired_deterministic_max"],
            acceptance["field_mean_error_relative_to_H1_maximum"],
        ),
        (
            field["aggregate_mean_rmse_relative_to_paired_deterministic_max"],
            acceptance["field_mean_error_relative_to_H1_maximum"],
        ),
        (
            field["primary_spread_skill_range"],
            acceptance["field_spread_skill_primary_range"],
        ),
        (
            field["remaining_spread_skill_range"],
            acceptance["field_spread_skill_fifth_field_range"],
        ),
        (
            spectral["member_expected_power_ratio_range"],
            acceptance["material_power_ratio_range"],
        ),
        (
            spectral["ensemble_mean_realization_coherence_min"],
            acceptance["material_realization_coherence_minimum"],
        ),
        (
            spectral["cross_phase_error_degrees_max"],
            acceptance["material_cross_phase_error_degrees_maximum"],
        ),
        (
            spectral["cross_coherence_absolute_change_max"],
            acceptance["material_cross_coherence_change_maximum"],
        ),
        (
            spectral["material_calibration_spread_skill_range"],
            acceptance["joint_projection_spread_skill_range"],
        ),
        (
            transport["strict_faces"]["relative_l2_max"],
            acceptance["strict_face_transport_relative_L2_maximum"],
        ),
        (
            transport["separatrix"]["relative_l2_max"],
            acceptance["separatrix_transport_relative_L2_maximum"],
        ),
    )
    if any(observed != expected for observed, expected in comparisons):
        raise ValueError("B5 summary and inherited B3 thresholds differ")
    monte_carlo = inherited_b3_gates.get("monte_carlo", {})
    if (
        monte_carlo.get("relative_difference_max")
        != acceptance["Monte_Carlo_M16_vs_M32_relative_tolerance"]
        or monte_carlo.get("absolute_floor")
        != acceptance["Monte_Carlo_absolute_tolerance"]
    ):
        raise ValueError("B5 inherited Monte Carlo thresholds differ")
    return numerical


def _relabel_parent_h1(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("name"), str):
                value["name"] = value["name"].replace("paired_H2", "parent_H1")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


def _material_band_counts(materiality: Mapping[str, Any]) -> tuple[int, int]:
    field_count = sum(
        item.get("material") is True
        for field in B2_FIELDS
        for item in materiality.get("fields", {})
        .get(field, {})
        .get("bands", {})
        .values()
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


def evaluate_b5_integrity(
    *,
    result: Mapping[str, Any],
    score: Mapping[str, Any],
    training: Mapping[str, Any],
    generation: Mapping[str, Any],
    comparator: Mapping[str, Any],
    manifest: Mapping[str, Any],
    training_wandb: Mapping[str, Any],
    evaluation_wandb: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate B5 provenance, closure, and truth-separation prerequisites."""

    book = CheckBook()
    book.boolean(
        "integrity.manifest.status",
        manifest.get("protocol_status")
        == (
            "frozen_after_passing_job_6901469_before_B5_full_training_"
            "validation_or_evaluation_implementation"
        ),
    )
    for name, expected in (
        ("development_run", "85604"),
        ("sequestered_run", "85606"),
        ("held_out_85606_access_allowed", False),
    ):
        book.boolean(f"integrity.manifest.{name}", manifest.get(name) == expected)
    data = manifest.get("data", {})
    book.boolean(
        "integrity.manifest.fields",
        data.get("fields") == ["Ne", "Pe", "Pi", "phi", "Vi"],
    )
    book.boolean(
        "integrity.manifest.no_time_input",
        data.get("absolute_time_input_allowed") is False,
    )
    book.boolean("integrity.manifest.zperiod_5", data.get("zperiod") == 5)
    book.boolean("integrity.manifest.n_eq_5k", data.get("mode_mapping") == "n=5k")

    book.boolean("integrity.training.scope", training.get("scope") == B5_TRAINING_SCOPE)
    for name, expected in (
        ("status", "training_completed_checkpoint_selected"),
        ("completed_epochs", 100),
        ("completed_optimizer_steps", 10_800),
        ("EMA_updates", 10_800),
        ("candidate_count", 20),
        ("checkpoint_reload_bitwise_exact", True),
        ("all_losses_and_gradients_finite", True),
        ("parameter_count", 11_604_709),
        ("physics_derived_loss_used", False),
        ("physics_metric_used_for_checkpoint_selection", False),
        ("sampled_forecast_metric_used_for_checkpoint_selection", False),
        ("target_truth_used_as_condition", False),
        ("absolute_time_used_as_condition", False),
        ("guard_frames_read", False),
        ("held_out_85606_read", False),
        ("scientific_forecast_generated", False),
        ("scientific_acceptance_evaluated", False),
    ):
        book.boolean(f"integrity.training.{name}", training.get(name) == expected)
    book.boolean(
        "integrity.training.selected_candidate",
        training.get("selected_completed_epoch") in range(5, 101, 5)
        and training.get("selected_optimizer_step")
        == training.get("selected_completed_epoch", -1) * 108,
    )

    book.boolean(
        "integrity.evaluation.scope", result.get("scope") == B5_EVALUATION_SCOPE
    )
    for name, expected in (
        ("status", "completed_pending_frozen_acceptance_gate"),
        ("scientific_authority", True),
        ("bounded_non_scientific_smoke", False),
        ("development_run", "85604"),
        ("held_out_85606_read", False),
        ("guard_frames_read", False),
        ("target_truth_used_during_forecast_generation", False),
        ("truth_opened_only_after_forecast_hash", True),
        ("absolute_time_used_as_model_input", False),
        ("target_frames", [498, 624]),
        ("target_count", 126),
        ("ensemble_members", 32),
        ("member_prefixes_regenerated", False),
        ("posthoc_calibration_applied", False),
        ("physics_derived_training_loss_used", False),
        ("full_evaluation_preconditions_passed", True),
        ("scientific_acceptance_evaluated", False),
        ("O3_launch_allowed", False),
        ("assimilation_allowed", False),
        ("diagnostic_ranking_allowed", False),
        ("seed", 1701),
    ):
        book.boolean(f"integrity.evaluation.{name}", result.get(name) == expected)
    book.boolean(
        "integrity.evaluation.checkpoint_matches_training",
        result.get("selected_checkpoint", {}).get("sha256")
        == training.get("artifacts", {}).get("selected_checkpoint", {}).get("sha256"),
    )
    history = result.get("training_history_audit", {})
    book.boolean(
        "integrity.evaluation.complete_history",
        history.get("epochs") == 100
        and history.get("optimizer_steps") == 10_800
        and history.get("candidate_count") == 20
        and history.get("earliest_validation_minimum_completed_epoch")
        == training.get("selected_completed_epoch")
        and history.get("finite") is True,
    )
    book.boolean(
        "integrity.evaluation.manifest_lock",
        result.get("evaluation_manifest", {}).get("sha256") == B5_MANIFEST_SHA256,
    )
    book.boolean(
        "integrity.evaluation.protocol_lock",
        result.get("evaluation_protocol", {}).get("sha256") == B5_PROTOCOL_SHA256,
    )
    book.boolean(
        "integrity.evaluation.event_threshold_lock",
        result.get("event_threshold_result", {}).get("sha256")
        == B5_EVENT_THRESHOLD_SHA256,
    )
    scientific_seed = result.get("scientific_sampler_seed_bank", {})
    book.boolean(
        "integrity.evaluation.scientific_seed_bank",
        scientific_seed.get("seed") == 67_532
        and scientific_seed.get("shape") == [126, 32]
        and scientific_seed.get("sha256") == B5_SCIENTIFIC_SEED_SHA256,
    )

    book.boolean(
        "integrity.generation.scope", generation.get("scope") == B5_GENERATION_SCOPE
    )
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
        "integrity.generation.canonical_shape",
        generation.get("forecast", {}).get("shape") == [126, 32, 1, 5, 64, 32, 88],
    )
    bank = generation.get("scientific_sampler_seed_bank", {})
    book.boolean(
        "integrity.generation.seed_bank",
        bank.get("sha256") == B5_SCIENTIFIC_SEED_SHA256
        and bank.get("complete_M32_generated_once") is True
        and bank.get("independent_of_checkpoint_selection_noise") is True,
    )
    book.boolean(
        "integrity.generation.sampler",
        generation.get("inference", {}).get("kind")
        == "EDM_probability_flow_ODE_Heun_residual_sampling"
        and generation.get("inference", {}).get("sampler_steps") == 18
        and generation.get("inference", {}).get("network_evaluations_per_member") == 35,
    )

    book.boolean("integrity.score.scope", score.get("scope") == B5_SCORE_SCOPE)
    for name, expected in (
        ("bounded_non_scientific_smoke", False),
        ("development_run", "85604"),
        ("held_out_85606_read", False),
        ("guard_frames_read", False),
        ("target_truth_used_during_forecast_generation", False),
        ("truth_opened_only_after_forecast_was_closed_and_hash_verified", True),
        ("training_performed", False),
        ("physics_derived_training_loss_used", False),
        ("model_seed", 1701),
        ("target_frames", [498, 624]),
        ("target_count", 126),
        ("model_arm", "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI"),
        ("context_frames", 1),
    ):
        book.boolean(f"integrity.score.{name}", score.get(name) == expected)
    book.boolean(
        "integrity.score.forecast_hash",
        score.get("forecast_artifact", {}).get("sha256")
        == result.get("forecast", {}).get("sha256"),
    )
    book.boolean(
        "integrity.score.metric_engine",
        score.get("metric_engine", {}).get("numerical_definitions_changed_for_B5")
        is False
        and score.get("metric_engine", {}).get("source_sha256")
        == result.get("metric_source_sha256"),
    )

    book.boolean(
        "integrity.comparator.scope",
        comparator.get("scope") == B5_COMPARATOR_SCOPE,
    )
    book.boolean(
        "integrity.comparator.parent_lock",
        comparator.get("forecast", {}).get("sha256")
        == "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
        and comparator.get("score", {}).get("sha256")
        == "ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593",
    )
    book.boolean(
        "integrity.comparator.uncompressed",
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
        "H1_field_comparator": len(
            comparator.get("field", {}).get("chronological_blocks", [])
        ),
        "H1_transport_comparator": len(
            comparator.get("transport", {}).get("chronological_blocks", [])
        ),
        "uncompressed_field_comparator": len(
            comparator.get("best_uncompressed", {})
            .get("field", {})
            .get("chronological_blocks", [])
        ),
    }
    for name, count in block_counts.items():
        book.boolean(f"integrity.six_blocks.{name}", count == 6)
    book.boolean("integrity.wandb.training", _wandb_finished(training_wandb))
    book.boolean(
        "integrity.wandb.training_epochs", training_wandb.get("epochs_logged") == 100
    )
    book.boolean("integrity.wandb.evaluation", _wandb_finished(evaluation_wandb))

    field_score = score.get("field_and_marginal_calibration", {})
    for region in ("eligible_union", *B2_PRIMARY_REGIONS):
        for field in B2_FIELDS:
            nonzero = (
                field_score.get("regions", {})
                .get(region, {})
                .get("fields", {})
                .get(field, {})
                .get("spread_integrity", {})
                .get("nonzero_spread")
            )
            book.boolean(f"integrity.nonzero_spread.{region}.{field}", nonzero is True)
    materiality = score.get("transport_event_thresholds", {}).get(
        "spectral_materiality", {}
    )
    field_band_count, cross_band_count = _material_band_counts(materiality)
    book.boolean("integrity.material_field_band_exists", field_band_count > 0)
    book.boolean("integrity.material_cross_band_exists", cross_band_count > 0)
    record = book.record()
    record["material_field_band_count"] = field_band_count
    record["material_cross_band_count"] = cross_band_count
    record["chronological_block_counts"] = block_counts
    return record


def _all_numeric_finite(families: Mapping[str, Mapping[str, Any]]) -> bool:
    checks = []
    for family in families.values():
        checks.extend(family.get("checks", []))
        for block in family.get("chronological_blocks", []):
            checks.extend(block.get("checks", []))
    numeric = [item for item in checks if item.get("kind") == "numeric"]
    return bool(numeric) and all(item.get("finite") is True for item in numeric)


def evaluate_b5_one_seed_acceptance(
    *,
    result: Mapping[str, Any],
    score: Mapping[str, Any],
    training: Mapping[str, Any],
    generation: Mapping[str, Any],
    comparator: Mapping[str, Any],
    manifest: Mapping[str, Any],
    inherited_b3_gates: Mapping[str, Any],
    training_wandb: Mapping[str, Any],
    evaluation_wandb: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen B5 gate without training, inference, or rescoring."""

    numerical = validate_inherited_b3_thresholds(
        b5_manifest=manifest,
        inherited_b3_gates=inherited_b3_gates,
    )
    materiality = score["transport_event_thresholds"]["spectral_materiality"]
    field = _relabel_parent_h1(
        evaluate_field_family(
            score["field_and_marginal_calibration"],
            comparator["field"],
            comparator["best_uncompressed"]["field"],
            numerical["field"],
        )
    )
    spectral = _relabel_parent_h1(
        evaluate_spectral_family(
            score["spectral_and_cross_field"],
            materiality,
            numerical["spectral"],
        )
    )
    transport = _relabel_parent_h1(
        evaluate_transport_family_event_eligible(
            score["memberwise_transport"],
            comparator["transport"],
            numerical["transport"],
        )
    )
    integrity = evaluate_b5_integrity(
        result=result,
        score=score,
        training=training,
        generation=generation,
        comparator=comparator,
        manifest=manifest,
        training_wandb=training_wandb,
        evaluation_wandb=evaluation_wandb,
    )
    families = {"field": field, "spectral": spectral, "transport": transport}
    complete = bool(
        integrity["passes"] and all(family["passes"] for family in families.values())
    )
    marginal_pass_joint_fail = bool(
        integrity["passes"]
        and field["passes"]
        and (not spectral["passes"] or not transport["passes"])
    )
    disposition = (
        "B5_one_step_gate_passed_write_O3_O4_protocol_only"
        if complete
        else "B5_marginal_calibration_insufficient_joint_transport_failure_stop"
        if marginal_pass_joint_fail
        else "B5_one_step_gate_failed_localize_without_retuning"
    )
    return {
        "schema_version": 1,
        "scope": "phase3_B5_residual_EDM_seed1701_frozen_one_step_acceptance_85604",
        "seed": 1701,
        "event_block_policy": EVENT_BLOCK_POLICY,
        "threshold_inheritance": {
            "source": "frozen_B3_complete_numerical_gate",
            "values_changed": False,
            "B5_summary_cross_checked": True,
        },
        "integrity": integrity,
        "families": families,
        "all_required_numeric_metrics_finite": _all_numeric_finite(families),
        "passes_complete_one_seed_gate": complete,
        "marginal_field_family_passes_but_joint_physics_fails": (
            marginal_pass_joint_fail
        ),
        "disposition": disposition,
        "O3_protocol_may_be_written": complete,
        "O3_launch_allowed": False,
        "additional_seed_training_authorized": False,
        "held_out_85606_access_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
