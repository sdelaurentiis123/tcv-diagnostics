import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/phase2_model_dataset_85604.json"
PROTOCOL = ROOT / "paper0/protocol/PHASE2_MODEL_DATASET_PROTOCOL.md"
RESAMPLING_MANIFEST = (
    ROOT / "paper0/manifests/phase2_85604_resampling_sensitivity.json"
)
BOUNDARY_RESULT = (
    ROOT / "paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
)


def load(path: Path):
    return json.loads(path.read_text())


def test_protocol_hash_and_held_out_guard_are_frozen():
    manifest = load(MANIFEST)
    protocol_digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()

    assert manifest["protocol"]["sha256"] == protocol_digest
    assert manifest["development_run"] == "85604"
    assert manifest["sequestered_run"] == "85606"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["training_authorized"] is False
    assert manifest["integrity_gates"]["all_gates_must_pass_before_training"]


def test_source_locks_match_prior_audited_evidence():
    manifest = load(MANIFEST)
    resampling = load(RESAMPLING_MANIFEST)
    boundary = load(BOUNDARY_RESULT)

    new_native = [
        (item["path"], item["sha256"])
        for item in manifest["sources"]["native_81_well"]
    ]
    old_native = [
        (item["path"], item["sha256"])
        for item in resampling["sources"]["native_81"]
    ]
    assert new_native == old_native

    new_legacy = [
        (item["path"], item["sha256"])
        for item in manifest["sources"]["legacy_z88_transform_oracle"]
    ]
    old_legacy = [
        (item["path"], item["sha256"])
        for item in resampling["sources"]["legacy_c5t_88"]
    ]
    assert new_legacy == old_legacy

    source = manifest["sources"]["potential_boundary"]
    evidence = boundary["external_artifacts"]["extraction_record"]
    assert source["extraction_record_path"] == evidence["path"]
    assert source["extraction_record_sha256"] == evidence["sha256"]
    assert source["source_job_id"] == boundary["slurm_job_id"] == 6893033


def test_field_union_and_state_views_are_exact():
    manifest = load(MANIFEST)
    expected_union = ["Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "phi", "Vi"]
    assert manifest["data"]["volume_fields"] == expected_union

    views = manifest["state_views_enabled_after_dataset_gate"]
    assert views["E6B-H1"] == {
        "context_frames": 1,
        "volume_fields": ["Ne", "Pe", "Pi", "NVe", "NVi", "Vort"],
        "boundary_field": "Bphi",
    }
    assert views["C5P-H2"]["context_frames"] == 2
    assert views["C5P-H2"]["volume_fields"] == [
        "Ne",
        "Pe",
        "Pi",
        "phi",
        "Vi",
    ]
    assert views["C5P-H1"]["context_frames"] == 1
    assert views["C5P-H1"]["volume_fields"] == views["C5P-H2"]["volume_fields"]


def test_shards_cover_every_frame_once_and_have_frozen_shapes():
    manifest = load(MANIFEST)
    intervals = manifest["output"]["shard_intervals"]

    assert intervals == [
        [0, 78],
        [78, 156],
        [156, 234],
        [234, 312],
        [312, 390],
        [390, 468],
        [468, 546],
        [546, 624],
    ]
    covered = [
        frame
        for start, stop in intervals
        for frame in range(start, stop)
    ]
    assert covered == list(range(624))
    assert manifest["output"]["volume_dataset_shape"] == [78, 64, 32, 88]
    assert manifest["output"]["volume_chunk_shape"] == [1, 64, 32, 88]
    assert manifest["output"]["boundary_dataset_shape"] == [78, 2, 32]
    assert manifest["output"]["overwrite_allowed"] is False


def test_normalization_uses_training_frames_only_and_preserves_pressure():
    manifest = load(MANIFEST)
    split = manifest["paper0_split"]
    normalization = manifest["normalization"]

    assert split["training_frames"] == [0, 432]
    assert split["guard_frames"] == [432, 496]
    assert split["validation_frames"] == [496, 624]
    assert split["normalization_fit_frames"] == split["training_frames"]
    assert normalization["volume_count_per_field"] == 432 * 64 * 32 * 88
    assert normalization["boundary_count_per_side"] == 432 * 32
    assert normalization["accumulator_dtype"] == "float64"
    assert normalization["transforms"]["Ne"] == {
        "name": "log_offset",
        "offset": 1e-6,
        "strictly_positive_argument_required": True,
    }
    assert normalization["transforms"]["Pi"]["name"] == "identity"
    assert manifest["data"]["retain_negative_Pi"] is True
    assert manifest["data"]["clipping_allowed"] is False


def test_toroidal_mapping_and_numeric_gates_are_explicit():
    manifest = load(MANIFEST)
    data = manifest["data"]
    gates = manifest["integrity_gates"]

    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5*k"
    assert data["native_nonnegative_k"] == [0, 40]
    assert data["output_padding_k"] == [41, 44]
    assert manifest["resampling"]["implementation"] == "scipy.signal.resample"
    assert gates["legacy_z88_bitwise_fields"] == ["Ne", "phi", "Vi"]
    assert gates["legacy_z88_bitwise_all_frames"] is True
    assert gates["field_round_trip_max_per_frame_relative_l2"] == 2e-6
    assert gates["normalization_recomputation_relative_tolerance"] == 1e-12
    assert gates["normalization_standard_deviations_strictly_positive"] is True
    assert gates["rocky_major_version_required"] == 9


def test_normalized_and_physical_time_are_not_conflated():
    manifest = load(MANIFEST)
    time = manifest["data"]["time"]
    gates = manifest["integrity_gates"]

    assert time["coordinate_name"] == "normalized_ion_cyclotron_time"
    assert time["first"] == 285000.0
    assert time["last"] == 471900.0
    assert time["normalized_step"] == 300.0
    physical = time["normalized_step"] / time["omega_ci_per_second"] * 1e6
    assert physical == time["physical_cadence_microseconds"]
    assert time["output_coordinate_is_normalized_not_physical"] is True
    assert gates["normalized_step_matches_manifest"] is True
    assert (
        gates["physical_cadence_after_omega_ci_conversion_matches_manifest"]
        is True
    )


def test_protocol_math_source_is_renderable_markdown():
    text = PROTOCOL.read_text()

    assert "\r" not in text
    assert "\\[" in text
    assert "\\left[" in text
    assert "\\frac{2\\pi}{5}" in text
    assert "\\Omega_{ci}" in text
    assert "\\widehat{x}" in text
    assert "\\epsilon_{\\mathrm{rel}\\,2}" in text
    assert "\\" + chr(96) not in text
