"""Regression lock for completed non-scientific B2 smoke 6896402."""

from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase3_b2_ldm_gpu_smoke_6896402.json"


def test_b2_smoke_result_is_byte_exact_and_strictly_non_scientific() -> None:
    assert sha256_path(RESULT) == (
        "fa2b29665b4b39b60c9ce24c1e8b067ebc6165322d40bb8de169bf9492ae5360"
    )
    result = load_strict_json(RESULT)
    assert result["slurm_job_id"] == "6896402"
    assert result["paper0_commit"] == "d58b4cc261a901b69c772b01270f38a89deb042f"
    assert result["development_run"] == "85604"
    assert result["rocky9_cpu_suite_passed"] is True
    assert result["one_gpu_smoke_passed"] is True
    assert result["seed"] == 1701
    assert result["training_targets"] == 16
    assert result["epochs"] == 2
    assert result["optimizer_steps"] == 2
    assert result["scientific_result"] is False
    assert result["full_B2_training_authorized"] is False
    assert result["held_out_85606_read"] is False
    assert result["O3_launch_allowed"] is False
    assert result["probabilistic_evaluation_protocol_still_required"] is True


def test_b2_smoke_result_locks_real_sampler_shape_diversity_and_reload() -> None:
    result = load_strict_json(RESULT)
    probe = result["sampler_probe"]
    assert probe["canonical_forecast_shape"] == [1, 2, 1, 5, 64, 32, 88]
    assert probe["ensemble_size"] == 2
    assert probe["finite"] is True
    assert probe["nonzero_latent_diversity"] is True
    assert probe["nonzero_decoded_diversity"] is True
    assert probe["latent_member_rms_difference"] > 0.0
    assert probe["field_member_rms_difference"] > 0.0
    assert len(result["selected_checkpoint_sha256"]) == 64
