import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/phase2_matched_o1_o2_85604.json"
PROTOCOL = ROOT / "paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md"


def load_manifest():
    return json.loads(MANIFEST.read_text())


def test_protocol_and_evidence_hashes_are_locked():
    manifest = load_manifest()
    assert manifest["protocol_status"] == (
        "frozen_before_model_implementation_or_training"
    )
    assert manifest["protocol"]["path"] == str(PROTOCOL.relative_to(ROOT))
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == (
        manifest["protocol"]["sha256"]
    )
    for lock in manifest["evidence_locks"].values():
        path = ROOT / lock["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == lock["sha256"]


def test_held_out_and_scope_guards_remain_closed():
    manifest = load_manifest()
    assert manifest["development_run"] == "85604"
    assert manifest["sequestered_run"] == "85606"
    assert manifest["held_out_85606_access_allowed"] is False
    assert "85606_access" in manifest["forbidden_scope"]
    assert "stochastic_model_training" in manifest["forbidden_scope"]
    assert "O3_or_longer_rollout" in manifest["forbidden_scope"]
    assert all("85604" in item for item in manifest["authorized_scope"])


def test_common_split_and_state_arms_match_prior_protocol():
    manifest = load_manifest()
    dataset = manifest["dataset"]
    assert dataset["job_id"] == "6893525"
    assert dataset["zperiod"] == 5
    assert dataset["mode_mapping"] == "n=5k"
    assert dataset["training_frames"] == [0, 432]
    assert dataset["guard_frames"] == [432, 496]
    assert dataset["validation_frames"] == [496, 624]
    assert dataset["o2_training_target_frames"] == [2, 432]
    assert dataset["o2_validation_target_frames"] == [498, 624]
    assert dataset["absolute_time_input_allowed"] is False

    states = manifest["state_arms"]
    assert states["E6B-H1"]["volume_fields"] == [
        "Ne",
        "Pe",
        "Pi",
        "NVe",
        "NVi",
        "Vort",
    ]
    assert states["E6B-H1"]["boundary"]["shape"] == [2, 32]
    assert states["E6B-H1"]["o1_boundary_bypass"] is True
    assert states["E6B-H1"]["o2_truth_boundary_reset"] is False
    assert states["C5P-H2"]["context_frames"] == 2
    assert states["C5P-H1"]["context_frames"] == 1
    assert states["C5P-H2"]["volume_fields"] == [
        "Ne",
        "Pe",
        "Pi",
        "phi",
        "Vi",
    ]
    assert states["C5P-H2"]["shares_codec_with"] == "C5P-H1"
    assert states["C5P-H1"]["shares_codec_with"] == "C5P-H2"


def test_codec_ladder_is_prospective_and_covers_useful_modes():
    manifest = load_manifest()
    assert manifest["seeds"] == [1701, 1702, 1703]
    codecs = manifest["codec_candidates"]
    assert [item["name"] for item in codecs] == ["dcae_l20", "dcae_l10"]
    assert codecs[0]["latent_grid"] == [8, 4, 22]
    assert codecs[1]["latent_grid"] == [16, 8, 22]
    assert all(item["latent_grid"][-1] // 2 >= 7 for item in codecs)
    assert codecs[0]["latent_scalars"] < codecs[1]["latent_scalars"]
    assert codecs[0]["predictor_patch"] == [1, 1, 1]
    assert codecs[1]["predictor_patch"] == [2, 2, 1]
    assert manifest["codec_common_architecture"]["periodic_axes_xyz"] == [
        False,
        False,
        True,
    ]
    escalation = manifest["codec_escalation"]
    assert escalation["run_first"] == "dcae_l20"
    assert escalation["if_dcae_l10_any_seed_fails"] == "stop_before_O2"


def test_training_losses_are_data_only_and_budgets_are_matched():
    manifest = load_manifest()
    codec = manifest["codec_training"]
    assert codec["from_scratch"] is True
    assert codec["epochs"] == 200
    assert codec["examples_per_epoch"] == 432
    assert codec["effective_batch"] == 16
    assert codec["loss"] == "equal_channel_standardized_MAE"
    assert codec["physics_derived_loss_allowed"] is False
    assert codec["early_stopping"] is False

    o2 = manifest["o2_training"]
    assert o2["epochs"] == 200
    assert o2["targets_per_epoch"] == 430
    assert o2["gradient_accumulation"] == 16
    assert o2["physics_derived_loss_allowed"] is False
    assert o2["early_stopping"] is False
    assert manifest["o2_model"]["noise_features"] == 0
    assert manifest["o2_model"]["loss_applies_to_target_only"] is True


def test_references_and_seed_level_stop_go_rules_are_explicit():
    manifest = load_manifest()
    references = manifest["uncompressed_references"]
    assert references["persistence"] is True
    assert references["c5p_h2_linear_extrapolation"] is True
    assert references["toroidal_spectral_ar1"]["fit_region"] == [0, 432]
    assert references["toroidal_spectral_ar1"]["relative_ridge"] == 1e-8

    assert manifest["o1_gates"]["required_passing_blocks"] == 7
    assert manifest["o2_gate"]["all_three_seeds_must_pass"] is True
    assert manifest["o2_gate"]["required_passing_blocks"] == 5
    assert (
        manifest["o2_gate"][
            "must_beat_best_uncompressed_reference_in_aggregate_rmse_and_mae"
        ]
        is True
    )


def test_protocol_math_source_is_renderable_markdown():
    text = PROTOCOL.read_text()
    assert "\r" not in text
    assert "\\[" in text
    assert "\\mathcal L_{\\rm codec}" in text
    assert "\\widehat z_{t+1}" in text
    assert "\\widehat B_{\\phi,t+1}" in text
    assert "\\" + chr(96) not in text
