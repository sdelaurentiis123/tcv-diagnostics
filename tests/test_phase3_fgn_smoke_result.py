"""Regression lock for the completed bounded B3 FGN smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase3_b3_fgn_gpu_smoke_6898604.json"


def test_b3_smoke_result_is_byte_locked_and_passed() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "dbac54c033917abbfec7e380d96a0c9be93667ae58240b4403400b57c76e2808"
    )
    record = json.loads(RESULT.read_text())
    assert record["status"] == "passed"
    assert record["scope"] == "phase3_B3_FGN_H1_bounded_GPU_smoke"
    assert record["paper0_commit"] == (
        "fb89828f6837ce0568fe7fc565b931810da68262"
    )
    assert record["slurm_job_id"] == "6898604"
    assert record["development_run"] == "85604"
    assert record["held_out_85606_read"] is False
    assert record["scientific_result"] is False
    assert record["full_B3_training_authorized"] is False


def test_b3_smoke_parent_codec_reload_and_members_passed() -> None:
    record = json.loads(RESULT.read_text())
    assert record["preoptimization_parent_identity"] == {
        "bitwise_exact": True,
        "maximum_absolute_difference": 0.0,
        "target_frame_index": 498,
    }
    assert record["codec_bitwise_unchanged"] is True
    assert record["checkpoint_reload_bitwise_exact"] is True
    probe = record["member_probe"]
    assert probe["finite"] is True
    assert probe["nonzero_latent_diversity"] is True
    assert probe["nonzero_field_diversity"] is True
    assert probe["reload_latent_bitwise_exact"] is True
    assert probe["reload_forecast_bitwise_exact"] is True
    assert probe["canonical_forecast_shape"] == [1, 2, 1, 5, 64, 32, 88]
    assert probe["latent_member_rms_difference"] == 0.16488301753997803
    assert probe["field_member_rms_difference"] == 0.046771977096796036


def test_b3_smoke_artifact_and_wandb_provenance_are_exact() -> None:
    record = json.loads(RESULT.read_text())
    assert record["run_result"]["sha256"] == (
        "fccb26d5ee22d7bf8e716a3ac483263d0bce6fde32d4bdaadf8a015a6700ffd1"
    )
    assert record["artifact_index"]["sha256"] == (
        "8cd352368e40639d230b384d528a93c67d833098f9d9ff4ee1efa53cd3f20f62"
    )
    assert record["selected_checkpoint"]["sha256"] == (
        "0390d6f7b96497688b34e0d0aedf5ffeccacae59eacacdcd09438eabf47345de"
    )
    wandb = record["wandb"]
    assert wandb["required"] is True
    assert wandb["mode"] == "online"
    assert wandb["remote_presence_verified_after_finish"] is True
    assert wandb["remote_state_after_finish"] == "finished"
    assert wandb["epochs_logged"] == 2
    assert wandb["checkpoints_uploaded"] is False
