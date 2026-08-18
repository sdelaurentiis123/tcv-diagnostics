"""Regression locks for the frozen 85604 O2 scientific-evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tcv_diagnostics.o2_evaluation import O2_THRESHOLDS


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0/protocol/PHASE2_O2_SCIENTIFIC_EVALUATION_PROTOCOL.md"
MANIFEST = ROOT / "paper0/manifests/phase2_o2_scientific_evaluation_85604.json"
FREEZE = ROOT / "paper0/results/phase2_o2_training_freeze_6895637.json"
PROTOCOL_SHA256 = "61200fa6224a035f6f5b129eaa27e849ad3b55bb4c3901c751c3a3fbf7b65aa4"
FREEZE_SHA256 = "dd8951e39e60d1631866ebe7af7c4d529ad543daf211233369b8fec9936ee837"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records() -> tuple[dict, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    return manifest, freeze


def test_protocol_was_frozen_before_scientific_evaluation_and_keeps_85606_closed():
    manifest, _ = _records()
    assert _sha256(PROTOCOL) == PROTOCOL_SHA256
    assert manifest["protocol"]["sha256"] == PROTOCOL_SHA256
    assert manifest["status"] == "frozen_before_O2_scientific_evaluation"
    assert manifest["decision_timing"] == {
        "after_training_checkpoint_freeze": True,
        "before_reference_forecast_generation": True,
        "before_physics_metric_evaluation": True,
        "training_loss_already_observed": True,
        "training_loss_may_select_arm": False,
    }
    assert manifest["development_run"] == "85604"
    assert manifest["sequestered_run"] == "85606"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["next_stage"] == {
        "O2_scientific_evaluation_completed": False,
        "accepted_arms": [],
        "new_O3_protocol_may_be_frozen": False,
        "O3_launch_allowed": False,
        "stochastic_model_authorized": False,
        "held_out_85606_access_allowed": False,
    }


def test_manifest_locks_the_exact_completed_training_matrix():
    manifest, freeze = _records()
    assert _sha256(FREEZE) == FREEZE_SHA256
    assert manifest["evidence_locks"]["training_checkpoint_freeze"]["sha256"] == (
        FREEZE_SHA256
    )
    expected = [
        (
            int(run["run_index"]),
            run["arm"],
            int(run["context_frames"]),
            int(run["seed"]),
            int(run["selected_epoch"]),
            run["selected_checkpoint"]["path"],
            run["selected_checkpoint"]["sha256"],
            run["codec_checkpoint"]["path"],
            run["codec_checkpoint"]["sha256"],
        )
        for run in freeze["runs"]
    ]
    observed = [
        (
            int(run["run_index"]),
            run["arm"],
            int(run["context_frames"]),
            int(run["seed"]),
            int(run["selected_epoch"]),
            run["path"],
            run["sha256"],
            run["codec_path"],
            run["codec_sha256"],
        )
        for run in manifest["checkpoints"]
    ]
    assert observed == expected
    assert len(observed) == 6


def test_manifest_locks_leakage_safe_targets_references_and_mode_mapping():
    manifest, _ = _records()
    dataset = manifest["dataset"]
    assert dataset["training_frames"] == [0, 432]
    assert dataset["guard_frames"] == [432, 496]
    assert dataset["validation_target_frames"] == [498, 624]
    assert dataset["validation_blocks"] == [
        [498, 519],
        [519, 540],
        [540, 561],
        [561, 582],
        [582, 603],
        [603, 624],
    ]
    assert dataset["absolute_time_input_allowed"] is False
    assert dataset["future_truth_input_allowed"] is False
    assert dataset["zperiod"] == 5
    assert dataset["mode_mapping"] == "n=5k"
    assert manifest["forecast_artifact"]["context_loader_reads_target"] is False
    assert (
        manifest["forecast_artifact"][
            "target_truth_read_during_generation_allowed"
        ]
        is False
    )
    assert set(manifest["references"]) >= {
        "persistence",
        "spectral_ar1",
        "linear_extrapolation",
    }
    assert manifest["references"]["linear_extrapolation"]["applicable_arms"] == [
        "C5P-H2"
    ]
    assert manifest["references"]["spectral_ar1"]["validation_tuning_allowed"] is False


def test_manifest_gate_is_identical_to_the_frozen_implementation():
    manifest, _ = _records()
    gate = manifest["gate"]
    assert gate["minimum_fields_beating_persistence"] == O2_THRESHOLDS[
        "minimum_fields_beating_persistence"
    ]
    assert gate["maximum_field_persistence_rmse_ratio"] == O2_THRESHOLDS[
        "maximum_field_persistence_rmse_ratio"
    ]
    assert gate["spectral_power_ratio"] == list(
        O2_THRESHOLDS["spectral_power_ratio"]
    )
    assert gate["forecast_truth_coherence_min"] == O2_THRESHOLDS[
        "forecast_truth_coherence_min"
    ]
    assert gate["cross_phase_error_degrees_max"] == O2_THRESHOLDS[
        "cross_phase_error_degrees_max"
    ]
    assert gate["cross_coherence_change_max"] == O2_THRESHOLDS[
        "cross_coherence_change_max"
    ]
    assert gate["strict_faces"] == O2_THRESHOLDS["strict_faces"]
    assert gate["separatrix"] == O2_THRESHOLDS["separatrix"]
    assert gate["required_passing_blocks_per_component"] == O2_THRESHOLDS[
        "required_passing_blocks"
    ]
    assert gate["arm_acceptance_requires_all_three_seeds"] is True
    assert gate["seed_averaging_can_rescue_failure"] is False
