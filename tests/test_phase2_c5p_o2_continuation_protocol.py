import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/phase2_c5p_o2_continuation_85604.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest():
    return load_json(MANIFEST)


def test_continuation_is_hash_locked_after_o1_and_before_o2():
    manifest = load_manifest()
    assert manifest["protocol_status"] == (
        "frozen_after_complete_R2_O1_and_before_O2_implementation_or_training"
    )
    assert manifest["decision_timing"] == {
        "relative_to_O1": "outcome_informed_after_complete_R2_result",
        "relative_to_O2": "prospective_before_implementation_smoke_or_training",
    }
    protocol = ROOT / manifest["protocol"]["path"]
    assert sha256(protocol) == manifest["protocol"]["sha256"]
    for lock in manifest["evidence_locks"].values():
        path = ROOT / lock["path"]
        assert sha256(path) == lock["sha256"]


def test_historical_six_run_failure_is_not_rewritten():
    manifest = load_manifest()
    historical = manifest["historical_matrix_decision"]
    assert historical["must_remain_unchanged"] is True
    assert historical["R2_accepted"] is False
    assert historical["O2_launch_allowed"] is False
    assert historical["status"] == "fail"

    result_path = ROOT / manifest["evidence_locks"]["complete_R2_O1_result"]["path"]
    result = load_json(result_path)
    assert result["R2_accepted"] is False
    assert result["O2_launch_allowed"] is False
    assert result["status"] == "fail"
    assert result["all_six_complete"] is True


def test_representation_local_rule_selects_exactly_three_passing_c5p_seeds():
    manifest = load_manifest()
    result_path = ROOT / manifest["evidence_locks"]["complete_R2_O1_result"]["path"]
    result = load_json(result_path)
    c5p = [run for run in result["runs"] if run["family"] == "c5p"]
    e6b = [run for run in result["runs"] if run["family"] == "e6b"]

    assert [run["seed"] for run in c5p] == [1701, 1702, 1703]
    assert all(run["passes"] is True for run in c5p)
    assert [run["seed"] for run in e6b] == [1701, 1702, 1703]
    assert all(run["passes"] is False for run in e6b)

    selection = manifest["continuation_selection"]
    assert selection["unit"] == "representation"
    assert selection["selected_representation"] == "C5P-dcae_l10"
    assert selection["selected_seeds"] == [1701, 1702, 1703]
    assert selection["selected_pass_count"] == 3
    assert selection["selected_required_count"] == 3
    assert selection["competing_representation_failure_is_a_veto"] is False
    assert selection["rejected_representations"]["E6B-dcae_l10"]["pass_count"] == 0


def test_selected_checkpoint_locks_match_the_pre_metric_freeze():
    manifest = load_manifest()
    freeze_path = ROOT / manifest["evidence_locks"]["R2_checkpoint_freeze"]["path"]
    freeze = load_json(freeze_path)
    expected = {
        run["seed"]: (
            run["selected_epoch"],
            run["selected_checkpoint"]["path"],
            run["selected_checkpoint"]["sha256"],
        )
        for run in freeze["runs"]
        if run["family"] == "c5p"
    }
    selected = {
        run["seed"]: (
            run["selected_epoch"], run["path"], run["sha256"]
        )
        for run in manifest["codec"]["selected_checkpoints"]
    }
    assert selected == expected
    assert manifest["codec"]["trainable_during_O2"] is False
    assert manifest["codec"]["evaluation_mode_required"] is True


def test_only_c5p_h1_h2_on_85604_are_authorized():
    manifest = load_manifest()
    assert manifest["development_run"] == "85604"
    assert manifest["sequestered_run"] == "85606"
    assert manifest["held_out_85606_access_allowed"] is False
    assert set(manifest["arms"]) == {"C5P-H1", "C5P-H2"}
    assert manifest["arms"]["C5P-H1"]["context_frames"] == 1
    assert manifest["arms"]["C5P-H2"]["context_frames"] == 2
    assert "E6B_O2" in manifest["forbidden_scope"]
    assert "O3_or_longer_rollout" in manifest["forbidden_scope"]
    assert "85606_access" in manifest["forbidden_scope"]
    assert all("85604" in item for item in manifest["authorized_scope"])


def test_inherited_o2_settings_and_gates_are_unchanged():
    continuation = load_manifest()
    original_path = ROOT / continuation["evidence_locks"]["original_matched_manifest"]["path"]
    original = load_json(original_path)

    for key in (
        "family",
        "hidden_channels",
        "transformer_blocks",
        "attention_heads",
        "ffn_factor",
        "qk_normalization",
        "rope",
        "dropout",
        "noise_features",
        "activation_checkpointing",
        "tokens_per_frame",
        "target_slot_masked",
        "loss_applies_to_target_only",
        "prediction",
        "training_loss",
    ):
        assert continuation["model"][key] == original["o2_model"][key]

    for key in (
        "epochs",
        "targets_per_epoch",
        "microbatch",
        "gradient_accumulation",
        "optimizer",
        "learning_rate",
        "betas",
        "weight_decay",
        "gradient_clip",
        "warmup_epochs",
        "minimum_learning_rate",
        "scheduler",
        "checkpoint_selection",
        "early_stopping",
    ):
        assert continuation["training"][key] == original["o2_training"][key]

    key_map = {
        "validation_blocks": "validation_blocks",
        "targets_per_block": "targets_per_block",
        "must_beat_best_uncompressed_reference_in_aggregate_rmse_and_mae": (
            "must_beat_best_uncompressed_reference_in_aggregate_rmse_and_mae"
        ),
        "minimum_common_fields_beating_persistence": (
            "minimum_common_fields_beating_persistence"
        ),
        "maximum_per_field_persistence_ratio": "maximum_per_field_persistence_ratio",
        "spectral_power_ratio": "spectral_power_ratio",
        "forecast_truth_coherence_min": "forecast_truth_coherence_min",
        "cross_phase_degrees_max": "cross_phase_degrees_max",
        "cross_coherence_change_max": "cross_coherence_change_max",
        "required_passing_blocks": "required_passing_blocks",
    }
    for continuation_key, original_key in key_map.items():
        assert continuation["o2_gate"][continuation_key] == original["o2_gate"][original_key]
    assert continuation["o2_gate"]["strict_face_transport"] == original["o2_gate"]["strict_face_transport"]
    assert continuation["o2_gate"]["separatrix_transport"] == original["o2_gate"]["separatrix_transport"]
    assert continuation["o2_gate"]["arm_acceptance_requires_all_three_seeds"] is True
    assert continuation["o2_gate"]["seed_averaging_can_rescue_failure"] is False


def test_full_runs_require_rocky9_clean_commit_and_online_wandb():
    manifest = load_manifest()
    assert manifest["wandb"]["project"] == "tcv-diagnostics-paper0"
    assert manifest["wandb"]["full_run_mode"] == "online"
    assert manifest["wandb"]["successful_initialization_required_before_training"] is True
    assert manifest["implementation_gate"]["rocky9_only"] is True
    assert manifest["implementation_gate"]["clean_exact_commit_required"] is True
    assert manifest["implementation_gate"]["full_array_run_count"] == 6
    assert manifest["next_stage_authorized"] is False
