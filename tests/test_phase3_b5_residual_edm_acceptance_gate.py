"""Known-answer tests for the frozen B5 one-seed acceptance reduction."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import tcv_diagnostics.b5_residual_edm_acceptance_gate as gate
from tcv_diagnostics.b2_field_metrics import B2_FIELDS, B2_PRIMARY_REGIONS


ROOT = Path(__file__).resolve().parents[1]
B5_MANIFEST = json.loads(
    (
        ROOT / "paper0/manifests/phase3_b5_full_training_evaluation_85604.json"
    ).read_text()
)
B3_GATES = json.loads(
    (ROOT / "paper0/manifests/phase3_b3_full_evaluation_85604.json").read_text()
)["gates"]


def wandb(*, epochs: int | None = None) -> dict:
    record = {
        "required": True,
        "mode": "online",
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": "finished",
        "local_artifacts_are_scientific_authority": True,
    }
    if epochs is not None:
        record["epochs_logged"] = epochs
    return record


def integrity_inputs() -> dict:
    selected_sha = "a" * 64
    forecast_sha = "b" * 64
    sources = {"metric": "c" * 64}
    training = {
        "scope": gate.B5_TRAINING_SCOPE,
        "status": "training_completed_checkpoint_selected",
        "completed_epochs": 100,
        "completed_optimizer_steps": 10_800,
        "EMA_updates": 10_800,
        "candidate_count": 20,
        "checkpoint_reload_bitwise_exact": True,
        "all_losses_and_gradients_finite": True,
        "parameter_count": 11_604_709,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "sampled_forecast_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "scientific_acceptance_evaluated": False,
        "selected_completed_epoch": 50,
        "selected_optimizer_step": 5_400,
        "artifacts": {"selected_checkpoint": {"sha256": selected_sha}},
    }
    generation = {
        "scope": gate.B5_GENERATION_SCOPE,
        "bounded_non_scientific_smoke": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "member_interaction": False,
        "member_prefixes_regenerated": False,
        "posthoc_calibration": False,
        "target_frames": [498, 624],
        "target_count": 126,
        "forecast": {"shape": [126, 32, 1, 5, 64, 32, 88]},
        "scientific_sampler_seed_bank": {
            "sha256": gate.B5_SCIENTIFIC_SEED_SHA256,
            "complete_M32_generated_once": True,
            "independent_of_checkpoint_selection_noise": True,
        },
        "inference": {
            "kind": "EDM_probability_flow_ODE_Heun_residual_sampling",
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
        },
    }
    field_regions = {
        region: {
            "fields": {
                field: {"spread_integrity": {"nonzero_spread": True}}
                for field in B2_FIELDS
            }
        }
        for region in ("eligible_union", *B2_PRIMARY_REGIONS)
    }
    score = {
        "scope": gate.B5_SCORE_SCOPE,
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
        "model_arm": "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI",
        "context_frames": 1,
        "forecast_artifact": {"sha256": forecast_sha},
        "metric_engine": {
            "numerical_definitions_changed_for_B5": False,
            "source_sha256": sources,
        },
        "field_and_marginal_calibration": {
            "regions": field_regions,
            "chronological_blocks_eligible_union": [{} for _ in range(6)],
        },
        "spectral_and_cross_field": {"chronological_blocks": [{} for _ in range(6)]},
        "memberwise_transport": {"chronological_blocks": [{} for _ in range(6)]},
        "transport_event_thresholds": {
            "spectral_materiality": {
                "fields": {"Ne": {"bands": {"k1_3": {"material": True}}}},
                "cross_fields": {"Ne_phi": {"bands": {"k1_3": {"material": True}}}},
            }
        },
    }
    result = {
        "scope": gate.B5_EVALUATION_SCOPE,
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
        "full_evaluation_preconditions_passed": True,
        "scientific_acceptance_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "seed": 1701,
        "selected_checkpoint": {"sha256": selected_sha},
        "training_history_audit": {
            "epochs": 100,
            "optimizer_steps": 10_800,
            "candidate_count": 20,
            "earliest_validation_minimum_completed_epoch": 50,
            "finite": True,
        },
        "evaluation_manifest": {"sha256": gate.B5_MANIFEST_SHA256},
        "evaluation_protocol": {"sha256": gate.B5_PROTOCOL_SHA256},
        "event_threshold_result": {"sha256": gate.B5_EVENT_THRESHOLD_SHA256},
        "scientific_sampler_seed_bank": {
            "seed": 67_532,
            "shape": [126, 32],
            "sha256": gate.B5_SCIENTIFIC_SEED_SHA256,
        },
        "forecast": {"sha256": forecast_sha},
        "metric_source_sha256": sources,
    }
    comparator = {
        "scope": gate.B5_COMPARATOR_SCOPE,
        "forecast": {
            "sha256": "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
        },
        "score": {
            "sha256": "ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593"
        },
        "field": {"chronological_blocks": [{} for _ in range(6)]},
        "transport": {"chronological_blocks": [{} for _ in range(6)]},
        "best_uncompressed": {
            "name": "training_only_toroidal_spectral_AR1",
            "field": {"chronological_blocks": [{} for _ in range(6)]},
        },
    }
    return {
        "result": result,
        "score": score,
        "training": training,
        "generation": generation,
        "comparator": comparator,
        "manifest": B5_MANIFEST,
        "training_wandb": wandb(epochs=100),
        "evaluation_wandb": wandb(),
    }


def family(passes: bool) -> dict:
    return {
        "passes": passes,
        "checks": [{"kind": "numeric", "finite": True, "passes": passes}],
        "chronological_blocks": [],
    }


def test_B5_threshold_adapter_matches_every_frozen_summary_value() -> None:
    numerical = gate.validate_inherited_b3_thresholds(
        b5_manifest=B5_MANIFEST,
        inherited_b3_gates=B3_GATES,
    )
    assert numerical["field"]["primary_spread_skill_range"] == [0.8, 1.25]
    assert numerical["spectral"]["cross_phase_error_degrees_max"] == 20.0
    assert numerical["transport"]["separatrix"]["correlation_min"] == 0.8
    drifted = deepcopy(B5_MANIFEST)
    drifted["acceptance"]["material_power_ratio_range"] = [0.5, 2.0]
    with pytest.raises(ValueError, match="summary"):
        gate.validate_inherited_b3_thresholds(
            b5_manifest=drifted,
            inherited_b3_gates=B3_GATES,
        )


def test_B5_integrity_known_answer_passes_and_detects_time_leak() -> None:
    inputs = integrity_inputs()
    record = gate.evaluate_b5_integrity(**inputs)
    assert record["passes"] is True
    assert record["material_field_band_count"] == 1
    assert record["material_cross_band_count"] == 1
    leaked = deepcopy(inputs)
    leaked["result"]["absolute_time_used_as_model_input"] = True
    failed = gate.evaluate_b5_integrity(**leaked)
    assert failed["passes"] is False
    assert any(
        item["name"] == "integrity.evaluation.absolute_time_used_as_model_input"
        and item["passes"] is False
        for item in failed["checks"]
    )


def test_B5_complete_gate_and_marginal_joint_failure_dispositions(monkeypatch) -> None:
    inputs = integrity_inputs()
    monkeypatch.setattr(gate, "evaluate_b5_integrity", lambda **kwargs: family(True))
    monkeypatch.setattr(gate, "evaluate_field_family", lambda *args: family(True))
    monkeypatch.setattr(gate, "evaluate_spectral_family", lambda *args: family(True))
    monkeypatch.setattr(
        gate, "evaluate_transport_family_event_eligible", lambda *args: family(True)
    )
    result = gate.evaluate_b5_one_seed_acceptance(
        **inputs,
        inherited_b3_gates=B3_GATES,
    )
    assert result["passes_complete_one_seed_gate"] is True
    assert result["O3_protocol_may_be_written"] is True
    assert result["O3_launch_allowed"] is False

    monkeypatch.setattr(gate, "evaluate_spectral_family", lambda *args: family(False))
    failed = gate.evaluate_b5_one_seed_acceptance(
        **inputs,
        inherited_b3_gates=B3_GATES,
    )
    assert failed["passes_complete_one_seed_gate"] is False
    assert failed["marginal_field_family_passes_but_joint_physics_fails"] is True
    assert failed["disposition"] == (
        "B5_marginal_calibration_insufficient_joint_transport_failure_stop"
    )
    assert failed["O3_protocol_may_be_written"] is False
