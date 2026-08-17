import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "paper0"
    / "manifests"
    / "phase2_conditional_transient_state_85604.json"
)
PROTOCOL = (
    ROOT
    / "paper0"
    / "protocol"
    / "PHASE2_CONDITIONAL_TRANSIENT_STATE_PROTOCOL.md"
)


def load_manifest():
    return json.loads(MANIFEST.read_text())


def test_conditional_transient_partition_and_common_targets_are_frozen():
    manifest = load_manifest()
    temporal = manifest["temporal_protocol"]

    assert temporal["stationarity_screen_status"] == "failed_and_not_reclassified"
    assert temporal["training_frames"] == [0, 432]
    assert temporal["guard_frames"] == [432, 496]
    assert temporal["validation_frames"] == [496, 624]
    assert temporal["common_one_step_training_target_frames"] == [2, 432]
    assert temporal["common_one_step_validation_target_frames"] == [498, 624]
    assert temporal["training_target_count"] == 430
    assert temporal["validation_target_count"] == 126
    assert temporal["absolute_frame_or_time_input_allowed"] is False
    assert temporal["physical_independence_claimed"] is False

    train_start, train_stop = temporal["training_frames"]
    guard_start, guard_stop = temporal["guard_frames"]
    valid_start, valid_stop = temporal["validation_frames"]
    assert train_stop == guard_start
    assert guard_stop == valid_start
    assert train_start < train_stop < valid_start < valid_stop


def test_exact_and_pragmatic_state_candidates_are_explicit():
    states = load_manifest()["state_candidates"]

    exact = states["exact_primary"]
    assert exact["name"] == "E6B-H1"
    assert exact["context_frames"] == 1
    assert exact["volume_fields"] == ["Ne", "Pe", "Pi", "NVe", "NVi", "Vort"]
    assert exact["boundary_field"]["shape"] == [2, 32]
    assert exact["interior_phi_policy"] == "exact_hash_locked_elliptic_reconstruction"

    pragmatic = states["pragmatic_primary"]
    assert pragmatic["name"] == "C5P-H2"
    assert pragmatic["context_frames"] == 2
    assert pragmatic["volume_fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert pragmatic["predictive_sufficiency_claimed_before_test"] is False

    control = states["required_history_ablation"]
    assert control["name"] == "C5P-H1"
    assert control["context_frames"] == 1
    assert control["uses_common_target_frames"] is True


def test_negative_pressure_and_held_out_guards_remain_active():
    manifest = load_manifest()
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["normalization"]["retain_negative_Pi"] is True
    assert manifest["normalization"]["clipping_allowed"] is False
    assert manifest["o1_boundary_policy"]["E6B_boundary_bypasses_volumetric_codec"]
    assert manifest["o1_boundary_policy"]["o2_boundary_bypass_allowed"] is False
    assert "stationary_climatology" in manifest["forbidden_claims"]


def test_evidence_hashes_match_the_tracked_records():
    manifest = load_manifest()
    for lock in manifest["evidence_locks"].values():
        path = ROOT / lock["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == lock["sha256"]


def test_protocol_math_source_is_renderable_markdown():
    text = PROTOCOL.read_text()
    assert "\r" not in text
    assert "\\[" in text
    assert "\\Delta t" in text
    assert "\\left[" in text
    assert "\\right]" in text
    assert "\\phi" in text
    assert "\\" + chr(96) not in text
