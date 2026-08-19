"""Regression lock for completed non-scientific B4 smoke job 6899469."""

from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase3_b4_pde_refiner_gpu_smoke_6899469.json"


def test_b4_smoke_result_is_byte_exact_passed_and_non_scientific() -> None:
    assert sha256_path(RESULT) == (
        "fd2b5465f612eb8da4943f6284e317145eff64b25346895137981ce3e3993eef"
    )
    result = load_strict_json(RESULT)
    assert result["slurm_job_id"] == "6899469"
    assert result["paper0_commit"] == (
        "2277e1b9d402a2b1627950f772b9f77a6a054f9e"
    )
    assert result["status"] == "passed"
    assert result["development_run"] == "85604"
    assert result["held_out_85606_read"] is False
    assert result["scientific_result"] is False
    assert result["full_B4_training_authorized"] is False
    assert result["next_permitted_action"] == (
        "write_separate_full_training_and_evaluation_protocol_only"
    )


def test_b4_smoke_locks_parent_codec_reload_axes_and_diversity() -> None:
    result = load_strict_json(RESULT)
    assert result["preoptimization_parent_identity"] == {
        "bitwise_exact": True,
        "maximum_absolute_difference": 0.0,
        "target_frame_index": 498,
    }
    assert result["codec_bitwise_unchanged"] is True
    assert result["checkpoint_reload_bitwise_exact"] is True
    probe = result["member_and_stage_probe"]
    assert probe["canonical_final_forecast_shape"] == [4, 2, 1, 5, 64, 32, 88]
    assert probe["canonical_stage_shape"] == [4, 2, 4, 5, 64, 32, 88]
    assert probe["finite"] is True
    assert probe["level0_shared_bitwise_across_members"] is True
    assert probe["nonzero_final_diversity_in_every_field"] is True
    assert probe["reload_forecast_bitwise_exact"] is True
    assert probe["reload_latent_bitwise_exact"] is True
    assert set(probe["final_member_RMS_difference_by_field"]) == {
        "Ne",
        "Pe",
        "Pi",
        "phi",
        "Vi",
    }
    assert all(
        value > 0.0
        for value in probe["final_member_RMS_difference_by_field"].values()
    )


def test_b4_four_target_scores_remain_mechanical_not_acceptance_evidence() -> None:
    result = load_strict_json(RESULT)
    validation = result["validation_decoded_stages"]["validation"]
    assert validation["target_count"] == 4
    assert validation["ensemble_members"] == 2
    assert validation["refinement_levels"] == [0, 1, 2, 3]
    assert validation["physics_metrics_used"] is False
    level_mae = validation["equal_channel_MAE_by_level"]
    assert level_mae == [
        0.04238718431442976,
        0.04535351218655705,
        0.04551513930782676,
        0.045528923906385896,
    ]
    # Two optimizer steps only establish executable mechanics. The observed
    # increase is retained exactly and cannot be promoted to an H-det result.
    assert level_mae[-1] > level_mae[0]
    assert result["scientific_result"] is False


def test_b4_smoke_wandb_run_finished_online() -> None:
    wandb = load_strict_json(RESULT)["wandb"]
    assert wandb["mode"] == "online"
    assert wandb["required"] is True
    assert wandb["epochs_logged"] == 2
    assert wandb["remote_presence_verified_after_finish"] is True
    assert wandb["remote_state_after_finish"] == "finished"
    assert wandb["run_url"].endswith("/runs/p0b4smoke-6899469-s1701")
