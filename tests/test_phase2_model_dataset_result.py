import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase2_model_dataset_6893525.json"
NORMALIZATION = (
    ROOT / "paper0/results/phase2_model_dataset_normalization_6893525.json"
)
READOUT = ROOT / "paper0/PHASE2_MODEL_DATASET_READOUT.md"
RESULT_SHA256 = "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18"
NORMALIZATION_SHA256 = (
    "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7"
)


def load(path: Path):
    return json.loads(path.read_text())


def test_tracked_artifacts_are_byte_identical_to_rusty_results():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    assert (
        hashlib.sha256(NORMALIZATION.read_bytes()).hexdigest()
        == NORMALIZATION_SHA256
    )


def test_execution_identity_scope_and_decision_are_non_overreaching():
    result = load(RESULT)
    assert result["paper0_commit"] == "929ed0cb2a861742bcab34101bc60fd53970d40c"
    assert result["slurm_job_id"] == 6893525
    assert result["development_run"] == "85604"
    assert result["held_out_85606_read"] is False
    assert result["training_performed"] is False
    assert result["execution"]["git"]["worktree_clean"] is True
    assert result["execution"]["os"]["actual_major"] == 9
    assert result["decision"] == {
        "dataset_gate_passed": True,
        "meaning": "verified_shared_engineering_representation_only",
        "next_required_gate": "committed_matched_O1_O2_model_protocol",
        "training_released": False,
    }


def test_dataset_has_complete_unique_ordered_coverage():
    result = load(RESULT)
    dataset = result["dataset"]
    assert dataset["frame_count"] == 624
    assert dataset["fields"] == [
        "Ne",
        "Pe",
        "Pi",
        "NVe",
        "NVi",
        "Vort",
        "phi",
        "Vi",
    ]
    assert len(dataset["shards"]) == 8
    covered = []
    for expected_index, shard in enumerate(dataset["shards"]):
        assert shard["index"] == expected_index
        assert shard["global_stop_exclusive"] - shard[
            "global_start_inclusive"
        ] == 78
        covered.extend(
            range(
                shard["global_start_inclusive"],
                shard["global_stop_exclusive"],
            )
        )
        assert len(shard["field_array_sha256"]) == 8
        assert len(shard["Bphi_array_sha256"]) == 64
        assert len(shard["sha256"]) == 64
        assert len(shard["partial_sha256"]) == 64
    assert covered == list(range(624))
    assert dataset["total_bytes"] == 3599761472


def test_time_and_all_structural_gates_pass():
    result = load(RESULT)
    time = result["time"]
    gates = result["gates"]
    assert time == {
        "coordinate": "normalized_ion_cyclotron_time",
        "first": 285000.0,
        "last": 471900.0,
        "normalized_step": 300.0,
        "omega_ci_per_second": 95788333.03066081,
        "physical_cadence_microseconds": 3.131905426352636,
    }
    for key in (
        "source_hash_preflight",
        "complete_unique_frame_coverage",
        "normalized_time_sequence_exact",
        "physical_cadence_conversion",
        "all_inputs_and_reopened_outputs_finite",
        "writer_echo_bitwise_exact",
        "legacy_z88_bitwise_all_frames",
        "Bphi_explicit_float32_cast_bitwise_exact",
        "clean_git_checkout",
        "rocky_major_version",
        "all_passed",
    ):
        assert gates[key] is True


def test_all_eight_round_trip_errors_pass_the_prospective_limit():
    gates = load(RESULT)["gates"]["field_round_trip"]
    expected = {
        "Ne": 1.5420411778074044e-7,
        "Pe": 1.6032685093155274e-7,
        "Pi": 1.5968112514544762e-7,
        "NVe": 1.9674525853459356e-7,
        "NVi": 1.5760109223463784e-7,
        "Vort": 1.9486527907344742e-7,
        "phi": 1.5751610579802045e-7,
        "Vi": 1.5237029140525836e-7,
    }
    assert set(gates) == set(expected)
    for field, value in expected.items():
        assert gates[field]["maximum_per_frame_relative_l2"] == value
        assert gates[field]["limit"] == 2e-6
        assert gates[field]["passed"] is True
        assert value <= gates[field]["limit"]


def test_normalization_is_training_only_complete_and_nondegenerate():
    result = load(RESULT)
    normalization = load(NORMALIZATION)
    assert result["dataset"]["normalization"]["sha256"] == NORMALIZATION_SHA256
    assert normalization["fit_frames"] == [0, 432]
    assert normalization["accumulator_dtype"] == "float64"
    assert normalization["held_out_85606_read"] is False
    assert normalization["records"]["Ne"]["transform"] == {
        "name": "log_offset",
        "offset": 1e-6,
        "strictly_positive_argument_required": True,
    }
    for field in ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "phi", "Vi"):
        record = normalization["records"][field]
        assert record["count"] == 77856768
        assert record["population_standard_deviation"] > 0.0
    for side in ("inner", "outer"):
        record = normalization["records"][f"Bphi/{side}"]
        assert record["count"] == 13824
        assert record["population_standard_deviation"] > 0.0
    for gate in result["gates"]["normalization_recomputation"].values():
        assert gate["passed"] is True
        assert gate["standard_deviation_strictly_positive"] is True
        assert all(gate["comparisons"].values())


def test_readout_math_is_renderable_and_preserves_scope():
    text = READOUT.read_text()
    assert "\r" not in text
    assert "\\[" in text
    assert "\\Omega_{ci}" in text
    assert "\\epsilon_{\\mathrm{rel}\\,2}" in text
    assert "\\" + chr(96) not in text
    assert "Training performed or released:** no" in text
    assert "does not show that a codec or transition model works" in text
