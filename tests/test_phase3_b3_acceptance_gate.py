"""Known-answer tests for the frozen B3 one-seed acceptance reduction."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from tcv_diagnostics import fgn_acceptance_gate as gate
from tcv_diagnostics.b2_field_metrics import B2_FIELDS, B2_PRIMARY_REGIONS


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "paper0/manifests/phase3_b3_full_evaluation_85604.json").read_text()
)


def _training() -> dict[str, object]:
    return {
        "scope": gate.B3_TRAINING_SCOPE,
        "completed_epochs": 100,
        "completed_optimizer_steps": 2700,
        "checkpoint_reload_bitwise_exact": True,
        "codec_bitwise_unchanged": True,
        "common_parameter_gradient_seen": True,
        "new_parameter_gradient_seen": True,
        "physics_derived_loss_used": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "held_out_85606_read": False,
        "scientific_result": False,
        "training_complete_is_scientific_acceptance": False,
        "probabilistic_scientific_gate_evaluated": False,
        "config": {
            "training_loss": (
                "equal_channel_decoded_standardized_field_fair_CRPS"
            )
        },
        "validation_noise_bank": {
            "seed": 31003,
            "shape": [126, 2, 32],
            "sha256": "selection-noise",
        },
        "member_probe": {"nonzero_field_diversity": True},
        "preoptimization_parent_identity": {"bitwise_exact": True},
        "deterministic_parent_load_audit": {"passed": True},
        "deterministic_parent": {
            "sha256": MANIFEST["deterministic_parent"]["checkpoint_sha256"]
        },
        "codec_checkpoint": {
            "sha256": MANIFEST["codec"]["checkpoint_sha256"],
            "trainable": False,
        },
        "latent_normalization": {
            "sha256": MANIFEST["codec"]["latent_normalization_sha256"],
            "refit": False,
        },
        "selected_epoch": 40,
        "selected_checkpoint": {"path": "/development/selected.pt", "sha256": "a" * 64},
    }


def _result(training: dict[str, object]) -> dict[str, object]:
    return {
        "scope": gate.B3_EVALUATION_SCOPE,
        "status": "completed_pending_frozen_acceptance_gate",
        "scientific_authority": True,
        "bounded_non_scientific_smoke": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "truth_opened_only_after_forecast_hash": True,
        "absolute_time_used_as_model_input": False,
        "target_frames": [498, 624],
        "target_count": 126,
        "ensemble_members": 32,
        "member_prefixes_regenerated": False,
        "posthoc_calibration_applied": False,
        "physics_derived_training_loss_used": False,
        "full_probabilistic_evaluation_preconditions_passed": True,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "seed": 1701,
        "selected_epoch": training["selected_epoch"],
        "selected_checkpoint": training["selected_checkpoint"],
        "training_history_audit": {
            "epochs": 100,
            "optimizer_steps": 2700,
            "selection_metric": (
                "fixed_M2_all126_equal_channel_decoded_field_fair_CRPS"
            ),
            "earliest_validation_minimum_epoch": training["selected_epoch"],
            "finite": True,
        },
        "checkpoint_selection_noise": {
            "sha256": training["validation_noise_bank"]["sha256"],
            "used_for_scientific_ensemble": False,
        },
        "evaluation_manifest": {"sha256": gate.B3_MANIFEST_SHA256},
        "evaluation_protocol": {"sha256": MANIFEST["protocol"]["sha256"]},
        "event_threshold_result": {"sha256": gate.B3_EVENT_THRESHOLD_SHA256},
        "scientific_noise": {
            "seed": 31032,
            "shape": [126, 32, 32],
            "sha256": gate.B3_SCIENTIFIC_NOISE_SHA256,
        },
        "forecast": {"sha256": "b" * 64},
        "metric_source_sha256": MANIFEST["locked_metric_sources"],
    }


def _generation() -> dict[str, object]:
    return {
        "scope": "B3_FGN_H1_one_step_M32_forecast_generation_85604",
        "bounded_non_scientific_smoke": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "member_interaction": False,
        "member_prefixes_regenerated": False,
        "target_frames": [498, 624],
        "target_count": 126,
        "forecast": {"shape": [126, 32, 1, 5, 64, 32, 88]},
        "inference": {
            "kind": "direct_functional_noise_single_pass",
            "network_evaluations_per_member": 1,
        },
        "raw_noise": {
            "independent_of_checkpoint_selection_noise": True,
            "complete_M32_generated_once": True,
            "sha256": gate.B3_SCIENTIFIC_NOISE_SHA256,
        },
    }


def _materiality() -> dict[str, object]:
    return {
        "fields": {
            field: {"bands": {"k1_3": {"material": field == "Ne"}}}
            for field in B2_FIELDS
        },
        "cross_fields": {
            "Ne-phi": {"bands": {"k1_3": {"material": True}}}
        },
    }


def _score(result: dict[str, object]) -> dict[str, object]:
    regions = {
        region: {
            "fields": {
                field: {"spread_integrity": {"nonzero_spread": True}}
                for field in B2_FIELDS
            }
        }
        for region in ("eligible_union", *B2_PRIMARY_REGIONS)
    }
    return {
        "scope": gate.B3_SCORE_SCOPE,
        "bounded_non_scientific_smoke": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "truth_opened_only_after_forecast_was_closed_and_hash_verified": True,
        "training_performed": False,
        "physics_derived_training_loss_used": False,
        "model_seed": 1701,
        "target_frames": [498, 624],
        "target_count": 126,
        "model_arm": "B3-FGN-H1",
        "context_frames": 1,
        "forecast_artifact": {"sha256": result["forecast"]["sha256"]},
        "metric_engine": {
            "numerical_definitions_changed_for_B3": False,
            "source_sha256": MANIFEST["locked_metric_sources"],
        },
        "transport_event_thresholds": {"spectral_materiality": _materiality()},
        "field_and_marginal_calibration": {
            "regions": regions,
            "chronological_blocks_eligible_union": [{} for _ in range(6)],
        },
        "spectral_and_cross_field": {
            "chronological_blocks": [{} for _ in range(6)]
        },
        "memberwise_transport": {"chronological_blocks": [{} for _ in range(6)]},
    }


def _comparator() -> dict[str, object]:
    parent = MANIFEST["comparators"]["primary_deterministic_parent"]
    return {
        "scope": gate.B3_COMPARATOR_SCOPE,
        "status": "completed_before_B3_scientific_acceptance",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "B3_forecasts_or_scores_read": False,
        "deterministic_model_retrained": False,
        "deterministic_checkpoint_reselected": False,
        "uncompressed_reference_reselected": False,
        "scientific_acceptance_evaluated": False,
        "seed": 1701,
        "arm": "C5P-H1",
        "forecast": {"sha256": parent["forecast_sha256"]},
        "score": {"sha256": parent["score_sha256"]},
        "field": {
            "context_frames": 1,
            "chronological_blocks": [{} for _ in range(6)],
        },
        "transport": {"chronological_blocks": [{} for _ in range(6)]},
        "best_uncompressed": {
            "name": "training_only_toroidal_spectral_AR1",
            "field": {"chronological_blocks": [{} for _ in range(6)]},
        },
    }


def _wandb(*, epochs: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "required": True,
        "mode": "online",
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": "finished",
    }
    if epochs is not None:
        record["epochs_logged"] = epochs
    return record


def _inputs() -> dict[str, object]:
    training = _training()
    result = _result(training)
    return {
        "result": result,
        "score": _score(result),
        "training": training,
        "generation": _generation(),
        "comparator": _comparator(),
        "manifest": MANIFEST,
        "training_wandb": _wandb(epochs=100),
        "evaluation_wandb": _wandb(),
    }


def _family(passes: bool, name: str = "example.paired_H2") -> dict[str, object]:
    return {
        "passes": passes,
        "checks": [
            {
                "name": name,
                "kind": "numeric",
                "finite": True,
                "passes": passes,
            }
        ],
        "chronological_blocks": [],
    }


def test_B3_gate_adapter_preserves_values_and_maps_parent_names() -> None:
    adapted = gate.adapt_b3_numerical_gates(MANIFEST["gates"])
    assert adapted["field"][
        "aggregate_mean_mae_relative_to_paired_deterministic_max"
    ] == MANIFEST["gates"]["field"][
        "aggregate_mean_mae_relative_to_parent_H1_max"
    ]
    assert adapted["transport"]["separatrix"][
        "fair_crps_better_than_paired_deterministic_required"
    ] == MANIFEST["gates"]["transport"]["separatrix"][
        "fair_crps_better_than_parent_H1_required"
    ]


def test_B3_integrity_accepts_known_answer_and_rejects_held_out_contamination() -> None:
    inputs = _inputs()
    record = gate.evaluate_b3_integrity(**inputs)
    assert record["passes"] is True
    assert record["material_field_band_count"] == 1
    assert record["material_cross_band_count"] == 1
    assert set(record["chronological_block_counts"].values()) == {6}

    contaminated = deepcopy(inputs)
    contaminated["result"]["held_out_85606_read"] = True
    record = gate.evaluate_b3_integrity(**contaminated)
    assert record["passes"] is False
    assert any(
        item["name"] == "integrity.evaluation.held_out_85606_read"
        and item["passes"] is False
        for item in record["checks"]
    )


def test_B3_one_seed_decision_requires_all_families(monkeypatch) -> None:
    inputs = _inputs()
    monkeypatch.setattr(gate, "evaluate_b3_integrity", lambda **_: {"passes": True})
    monkeypatch.setattr(gate, "evaluate_field_family", lambda *_: _family(True))
    monkeypatch.setattr(gate, "evaluate_spectral_family", lambda *_: _family(True))
    monkeypatch.setattr(
        gate, "evaluate_transport_family_event_eligible", lambda *_: _family(True)
    )
    accepted = gate.evaluate_b3_one_seed_acceptance(**inputs)
    assert accepted["passes_complete_one_seed_gate"] is True
    assert accepted["seed1702_1703_replication_protocol_may_be_written"] is True
    assert accepted["seed1702_1703_training_authorized"] is False
    names = [
        item["name"]
        for family in accepted["families"].values()
        for item in family["checks"]
    ]
    assert "example.parent_H1" in names
    assert all("paired_H2" not in name for name in names)

    monkeypatch.setattr(gate, "evaluate_spectral_family", lambda *_: _family(False))
    joint_failure = gate.evaluate_b3_one_seed_acceptance(**inputs)
    assert joint_failure["passes_complete_one_seed_gate"] is False
    assert joint_failure[
        "marginal_field_family_passes_but_joint_physics_fails"
    ] is True
    assert joint_failure["disposition"] == MANIFEST["decision_rule"][
        "marginal_calibration_pass_joint_physics_fail_disposition"
    ]


def test_B3_one_seed_integrity_failure_stops_replication(monkeypatch) -> None:
    inputs = _inputs()
    monkeypatch.setattr(gate, "evaluate_b3_integrity", lambda **_: {"passes": False})
    monkeypatch.setattr(gate, "evaluate_field_family", lambda *_: _family(True))
    monkeypatch.setattr(gate, "evaluate_spectral_family", lambda *_: _family(True))
    monkeypatch.setattr(
        gate, "evaluate_transport_family_event_eligible", lambda *_: _family(True)
    )
    rejected = gate.evaluate_b3_one_seed_acceptance(**inputs)
    assert rejected["passes_complete_one_seed_gate"] is False
    assert rejected["disposition"] == MANIFEST["decision_rule"][
        "point_skill_or_integrity_fail_disposition"
    ]
    assert rejected["O3_launch_allowed"] is False
    assert rejected["held_out_85606_access_allowed"] is False
