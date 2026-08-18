"""Frozen one-seed scientific acceptance reduction for B3-FGN-H1.

The numerical field, spectral, and transport checks are delegated to the
already-tested B2 gate implementation.  This module only maps the prospectively
frozen B3/H1 threshold names, adds B3-specific provenance checks, and applies
the one-seed decision rule frozen before the seed-1701 scientific evaluation.
"""

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


B3_EVALUATION_SCOPE = "B3_FGN_H1_full_probabilistic_evaluation_85604"
B3_SCORE_SCOPE = "B3_FGN_H1_truth_separated_probabilistic_scoring_85604"
B3_COMPARATOR_SCOPE = "phase3_B3_frozen_matched_H1_comparators_85604"
B3_TRAINING_SCOPE = "B3_FGN_H1_seed1701_full_training_85604"
B3_SCIENTIFIC_NOISE_SHA256 = (
    "1449777a61d40af49ccb3bd6bed5edcba0fd8afe24d113e6175218c04865aa9c"
)


def adapt_b3_numerical_gates(
    gates: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Map B3 parent-H1 names to the unchanged B2 numerical reducers."""

    required = {"field", "spectral", "transport"}
    if not required.issubset(gates):
        raise ValueError("B3 numerical gate families differ")
    field = deepcopy(dict(gates["field"]))
    transport = deepcopy(dict(gates["transport"]))
    field["aggregate_mean_mae_relative_to_paired_deterministic_max"] = field[
        "aggregate_mean_mae_relative_to_parent_H1_max"
    ]
    field["aggregate_mean_rmse_relative_to_paired_deterministic_max"] = field[
        "aggregate_mean_rmse_relative_to_parent_H1_max"
    ]
    field["fields_required_fair_crps_better_than_paired_deterministic"] = field[
        "fields_required_fair_crps_better_than_parent_H1"
    ]
    transport["separatrix"][
        "fair_crps_better_than_paired_deterministic_required"
    ] = transport["separatrix"][
        "fair_crps_better_than_parent_H1_required"
    ]
    return {
        "field": field,
        "spectral": deepcopy(dict(gates["spectral"])),
        "transport": transport,
    }


def _relabel_parent_h1(record: Mapping[str, Any]) -> dict[str, Any]:
    """Relabel inherited check names without changing values or thresholds."""

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
        for item in (
            materiality.get("fields", {}).get(field, {}).get("bands", {}).values()
        )
    )
    cross_count = sum(
        item.get("material") is True
        for pair in materiality.get("cross_fields", {}).values()
        for item in pair.get("bands", {}).values()
    )
    return field_count, cross_count


def evaluate_b3_integrity(
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
    """Evaluate every B3-specific non-numerical prerequisite explicitly."""

    book = CheckBook()
    book.boolean(
        "integrity.manifest_frozen_before_scientific_evaluation",
        manifest.get("protocol_status")
        == (
            "frozen_after_passing_B3_smoke_before_full_training_or_scientific_"
            "evaluation_implementation"
        ),
    )
    for name, expected in (
        ("development_run", "85604"),
        ("sequestered_run", "85606"),
        ("held_out_85606_access_allowed", False),
        ("full_training_authorized", True),
        ("probabilistic_evaluation_authorized", True),
    ):
        book.boolean(
            f"integrity.manifest.{name}", manifest.get(name) == expected
        )
    data = manifest.get("data", {})
    book.boolean(
        "integrity.manifest.physical_C5P_fields",
        data.get("fields") == ["Ne", "Pe", "Pi", "phi", "Vi"],
    )
    book.boolean(
        "integrity.manifest.context_has_no_time",
        data.get("absolute_time_input_allowed") is False,
    )
    book.boolean("integrity.manifest.zperiod_5", data.get("zperiod") == 5)
    book.boolean(
        "integrity.manifest.mode_mapping_n_eq_5k",
        data.get("mode_mapping") == "n=5k",
    )
    book.boolean(
        "integrity.manifest.truth_empty_event_rule_A016",
        manifest.get("transport", {}).get("truth_empty_event_rule")
        == "A016_explicit_not_applicable_after_integrity_checks",
    )

    book.boolean(
        "integrity.training.scope", training.get("scope") == B3_TRAINING_SCOPE
    )
    for name, expected in (
        ("completed_epochs", 100),
        ("completed_optimizer_steps", 2700),
        ("checkpoint_reload_bitwise_exact", True),
        ("codec_bitwise_unchanged", True),
        ("common_parameter_gradient_seen", True),
        ("new_parameter_gradient_seen", True),
        ("physics_derived_loss_used", False),
        ("target_truth_used_as_model_input", False),
        ("absolute_time_used_as_model_input", False),
        ("held_out_85606_read", False),
        ("scientific_result", False),
        ("training_complete_is_scientific_acceptance", False),
        ("probabilistic_scientific_gate_evaluated", False),
    ):
        book.boolean(
            f"integrity.training.{name}", training.get(name) == expected
        )
    training_config = training.get("config", {})
    book.boolean(
        "integrity.training.objective_is_decoded_field_fair_CRPS_M2",
        training_config.get("training_loss")
        == "equal_channel_decoded_standardized_field_fair_CRPS",
    )
    book.boolean(
        "integrity.training.fixed_validation_noise",
        training.get("validation_noise_bank", {}).get("seed") == 31003
        and training.get("validation_noise_bank", {}).get("shape") == [126, 2, 32],
    )
    book.boolean(
        "integrity.training.selected_checkpoint_member_probe_noncollapsed",
        training.get("member_probe", {}).get("nonzero_field_diversity") is True,
    )

    book.boolean(
        "integrity.evaluation.scope", result.get("scope") == B3_EVALUATION_SCOPE
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
        ("full_probabilistic_evaluation_preconditions_passed", True),
        ("probabilistic_scientific_gate_evaluated", False),
        ("O3_launch_allowed", False),
        ("assimilation_allowed", False),
        ("diagnostic_ranking_allowed", False),
        ("seed", 1701),
    ):
        book.boolean(
            f"integrity.evaluation.{name}", result.get(name) == expected
        )
    book.boolean(
        "integrity.evaluation.selected_epoch_matches_training",
        result.get("selected_epoch") == training.get("selected_epoch"),
    )
    book.boolean(
        "integrity.evaluation.checkpoint_matches_training",
        result.get("selected_checkpoint") == training.get("selected_checkpoint"),
    )
    book.boolean(
        "integrity.evaluation.scientific_noise",
        result.get("scientific_noise", {}).get("seed") == 31032
        and result.get("scientific_noise", {}).get("shape") == [126, 32, 32]
        and result.get("scientific_noise", {}).get("sha256")
        == B3_SCIENTIFIC_NOISE_SHA256
        and result.get("scientific_noise", {}).get("sha256")
        != training.get("validation_noise_bank", {}).get("sha256"),
    )

    book.boolean(
        "integrity.generation.scope",
        generation.get("scope")
        == "B3_FGN_H1_one_step_M32_forecast_generation_85604",
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
        ("target_frames", [498, 624]),
        ("target_count", 126),
    ):
        book.boolean(
            f"integrity.generation.{name}", generation.get(name) == expected
        )
    book.boolean(
        "integrity.generation.canonical_forecast_shape",
        generation.get("forecast", {}).get("shape")
        == [126, 32, 1, 5, 64, 32, 88],
    )
    book.boolean(
        "integrity.generation.direct_single_pass_per_member",
        generation.get("inference", {}).get("kind")
        == "direct_functional_noise_single_pass"
        and generation.get("inference", {}).get("network_evaluations_per_member")
        == 1,
    )
    book.boolean(
        "integrity.generation.independent_complete_M32_noise",
        generation.get("raw_noise", {}).get(
            "independent_of_checkpoint_selection_noise"
        )
        is True
        and generation.get("raw_noise", {}).get("complete_M32_generated_once")
        is True
        and generation.get("raw_noise", {}).get("sha256")
        == B3_SCIENTIFIC_NOISE_SHA256,
    )

    book.boolean("integrity.score.scope", score.get("scope") == B3_SCORE_SCOPE)
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
        ("model_arm", "B3-FGN-H1"),
        ("context_frames", 1),
    ):
        book.boolean(f"integrity.score.{name}", score.get(name) == expected)
    book.boolean(
        "integrity.score.forecast_hash_matches_evaluation",
        score.get("forecast_artifact", {}).get("sha256")
        == result.get("forecast", {}).get("sha256"),
    )
    book.boolean(
        "integrity.score.locked_metric_engine",
        score.get("metric_engine", {}).get("numerical_definitions_changed_for_B3")
        is False
        and score.get("metric_engine", {}).get("source_sha256")
        == manifest.get("locked_metric_sources")
        and result.get("metric_source_sha256")
        == manifest.get("locked_metric_sources"),
    )

    book.boolean(
        "integrity.comparator.scope",
        comparator.get("scope") == B3_COMPARATOR_SCOPE,
    )
    for name, expected in (
        ("status", "completed_before_B3_scientific_acceptance"),
        ("development_run", "85604"),
        ("held_out_85606_read", False),
        ("guard_frames_read", False),
        ("B3_forecasts_or_scores_read", False),
        ("deterministic_model_retrained", False),
        ("deterministic_checkpoint_reselected", False),
        ("uncompressed_reference_reselected", False),
        ("scientific_acceptance_evaluated", False),
        ("seed", 1701),
        ("arm", "C5P-H1"),
    ):
        book.boolean(
            f"integrity.comparator.{name}", comparator.get(name) == expected
        )
    parent = manifest.get("comparators", {}).get("primary_deterministic_parent", {})
    book.boolean(
        "integrity.comparator.parent_forecast_lock",
        comparator.get("forecast", {}).get("sha256") == parent.get("forecast_sha256"),
    )
    book.boolean(
        "integrity.comparator.parent_score_lock",
        comparator.get("score", {}).get("sha256") == parent.get("score_sha256"),
    )
    book.boolean(
        "integrity.comparator.H1_context",
        comparator.get("field", {}).get("context_frames") == 1,
    )
    book.boolean(
        "integrity.comparator.uncompressed_reference",
        comparator.get("best_uncompressed", {}).get("name")
        == "training_only_toroidal_spectral_AR1",
    )

    for label, tracking, expected_epochs in (
        ("training", training_wandb, 100),
        ("evaluation", evaluation_wandb, None),
    ):
        book.boolean(
            f"integrity.wandb.{label}.online_finished",
            tracking.get("required") is True
            and tracking.get("mode") == "online"
            and tracking.get("remote_presence_verified_after_finish") is True
            and tracking.get("remote_state_after_finish") == "finished",
        )
        if expected_epochs is not None:
            book.boolean(
                f"integrity.wandb.{label}.epochs_logged",
                tracking.get("epochs_logged") == expected_epochs,
            )

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
            book.boolean(
                f"integrity.nonzero_spread.{region}.{field}", nonzero is True
            )
    materiality = (
        score.get("transport_event_thresholds", {}).get("spectral_materiality", {})
    )
    field_band_count, cross_band_count = _material_band_counts(materiality)
    book.boolean(
        "integrity.material_field_band_exists", field_band_count > 0
    )
    book.boolean(
        "integrity.material_cross_band_exists", cross_band_count > 0
    )
    record = book.record()
    record["material_field_band_count"] = field_band_count
    record["material_cross_band_count"] = cross_band_count
    return record


def _all_numeric_finite(families: Mapping[str, Mapping[str, Any]]) -> bool:
    checks = []
    for family in families.values():
        checks.extend(family.get("checks", []))
        for block in family.get("chronological_blocks", []):
            checks.extend(block.get("checks", []))
    numeric = [item for item in checks if item.get("kind") == "numeric"]
    return bool(numeric) and all(item.get("finite") is True for item in numeric)


def evaluate_b3_one_seed_acceptance(
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
    """Apply the frozen B3 seed-1701 gate without training or rescoring."""

    if result.get("seed") != 1701 or score.get("model_seed") != 1701:
        raise ValueError("B3 one-seed gate requires seed 1701")
    if (
        manifest.get("decision_rule", {}).get("one_seed_complete_gate_must_pass")
        is not True
    ):
        raise ValueError("B3 one-seed decision rule differs")
    numerical = adapt_b3_numerical_gates(manifest.get("gates", {}))
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
            score["spectral_and_cross_field"], materiality, numerical["spectral"]
        )
    )
    transport = _relabel_parent_h1(
        evaluate_transport_family_event_eligible(
            score["memberwise_transport"],
            comparator["transport"],
            numerical["transport"],
        )
    )
    integrity = evaluate_b3_integrity(
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
        integrity["passes"] and all(item["passes"] for item in families.values())
    )
    marginal_pass_joint_fail = bool(
        integrity["passes"]
        and field["passes"]
        and (not spectral["passes"] or not transport["passes"])
    )
    if complete:
        disposition = manifest["decision_rule"]["passing_disposition"]
    elif marginal_pass_joint_fail:
        disposition = manifest["decision_rule"][
            "marginal_calibration_pass_joint_physics_fail_disposition"
        ]
    else:
        disposition = manifest["decision_rule"][
            "point_skill_or_integrity_fail_disposition"
        ]
    return {
        "schema_version": 1,
        "scope": "phase3_B3_FGN_H1_seed1701_frozen_one_step_acceptance_85604",
        "seed": 1701,
        "event_block_policy": EVENT_BLOCK_POLICY,
        "numerical_gate_engine": {
            "identity": "B2_gate_reducers_with_B3_parent_H1_key_adapter",
            "numerical_reducers_changed": False,
            "threshold_values_changed_during_name_adaptation": False,
            "check_labels_replaced": {"paired_H2": "parent_H1"},
        },
        "integrity": integrity,
        "families": families,
        "all_required_numeric_metrics_finite": _all_numeric_finite(families),
        "passes_complete_one_seed_gate": complete,
        "marginal_field_family_passes_but_joint_physics_fails": (
            marginal_pass_joint_fail
        ),
        "disposition": disposition,
        "seed1702_1703_replication_protocol_may_be_written": complete,
        "seed1702_1703_training_authorized": False,
        "O3_protocol_may_be_written": False,
        "O3_launch_allowed": False,
        "held_out_85606_access_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
