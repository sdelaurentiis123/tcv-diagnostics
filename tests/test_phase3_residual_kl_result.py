"""Regression locks for the authoritative residual-KL oracle result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase3_residual_kl_oracle_6904897.json"
WANDB = ROOT / "paper0/results/phase3_residual_kl_oracle_wandb_6904897.json"
ARTIFACTS = ROOT / "paper0/results/phase3_residual_kl_oracle_artifacts_6904897.txt"
INTERPRETATION = ROOT / "paper0/PHASE3_RESIDUAL_KL_INTERPRETATION.md"
EXPECTED_RESULT_SHA256 = (
    "4f0166308e71d308a960c004cb6f9c247f6e0d9de038d01df5f3a85037fb2879"
)
EXPECTED_WANDB_SHA256 = (
    "b8c55f07de2640cdde488e1b3baa9a270861453df143dc0a57a3b69c511cf160"
)
EXPECTED_ARTIFACTS_SHA256 = (
    "ba83da68080ae5a779dcd09be23bd1d2ad54156039d91048fa333374cca10673"
)
EXPECTED_INTERPRETATION_SHA256 = (
    "34533ecd4d3a226bc4863bf985d93fa92cc98456c02d4e7fb3467ee83a613c8e"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_residual_kl_result_is_byte_locked_and_85604_only() -> None:
    assert _digest(RESULT) == EXPECTED_RESULT_SHA256
    result = _load(RESULT)
    assert result["status"] == "completed_without_model_training_or_downstream_opening"
    assert result["scope"] == "residual_KL_representation_oracle_and_static_covariance_85604"
    assert result["development_run"] == "85604"
    assert result["paper0_commit"] == (
        "6e3469b1a37430a2493e5889f24c653f2f5f5418"
    )
    assert result["slurm_job_id"] == "6904897"
    assert result["held_out_85606_read"] is False
    assert result["guard_frames_read"] is False


def test_residual_kl_outcome_and_closed_boundaries_are_exact() -> None:
    result = _load(RESULT)
    assert result["primary_outcome"] == "K4_training_residual_span_does_not_transfer"
    assert result["positive_rank"] == 429
    assert result["selected_static_rank"] == 128
    assert result["tier_A_minimum_passing_rank"] is None
    assert result["tier_B_static_covariance_useful"] is False
    assert result["rank_zero_H1_plus_bias_maximum_absolute_difference"] == 0.0
    for key in (
        "checkpoint_loaded",
        "model_inference_performed",
        "model_training_performed",
        "optimizer_or_trainable_parameter_created",
        "physics_metric_used_as_training_loss",
        "O3_launched",
        "assimilation_performed",
        "diagnostic_ranking_performed",
    ):
        assert result[key] is False


def test_residual_kl_authoritative_scientific_and_table_hashes_are_locked() -> None:
    result = _load(RESULT)
    assert result["scientific_result"]["sha256"] == (
        "71be0e38285a06f98bd03138d3e1639a70d88665e698cbb4c96220e57dc991b7"
    )
    assert result["raw_sufficient_statistics"]["sha256"] == (
        "98a418b51aabdfee12537ed43c618d100a7625cdc4659eaed729db0d27353552"
    )
    assert result["raw_sufficient_statistics"]["array_count"] == 125
    assert result["tables"]["tier_A_rank_summary"]["sha256"] == (
        "4f006d4ba6d667fc57ddc4ffd158e07d7314d3c7db3b4ce29b5fbb289325aa57"
    )
    assert result["tables"]["tier_B_transport_covariance"]["sha256"] == (
        "d36b5a5d2dc9213d3cb20072bee26bbb797433be2d44a0192f0252601d0e74a6"
    )
    assert len(result["figures"]) == 12
    assert len({record["path"] for record in result["figures"]}) == 12


def test_residual_kl_wandb_record_is_byte_locked_and_compact_only() -> None:
    assert _digest(WANDB) == EXPECTED_WANDB_SHA256
    record = _load(WANDB)
    assert record["mode"] == "online"
    assert record["remote_state_after_finish"] == "finished"
    assert record["remote_presence_verified_after_finish"] is True
    assert record["spec"]["run_id"] == "p0reskl-6904897-s1701"
    for key in (
        "checkpoints_uploaded",
        "forecasts_uploaded",
        "simulation_fields_uploaded",
        "basis_arrays_uploaded",
        "raw_accumulators_uploaded",
        "figures_uploaded",
        "tables_uploaded",
    ):
        assert record[key] is False


def test_residual_kl_artifact_manifest_and_interpretation_are_locked() -> None:
    assert _digest(ARTIFACTS) == EXPECTED_ARTIFACTS_SHA256
    assert _digest(INTERPRETATION) == EXPECTED_INTERPRETATION_SHA256
    inventory = ARTIFACTS.read_text(encoding="utf-8")
    for digest in (
        EXPECTED_RESULT_SHA256,
        EXPECTED_WANDB_SHA256,
        "fcc32c3baaf0deb85fa55456612d3ab8beaf859af20b5ba86f94233c15e0dbbc",
        "71be0e38285a06f98bd03138d3e1639a70d88665e698cbb4c96220e57dc991b7",
    ):
        assert digest in inventory
    interpretation = INTERPRETATION.read_text(encoding="utf-8")
    assert "K4_training_residual_span_does_not_transfer" in interpretation
    assert "22.34%" in interpretation
    assert "Run 85606 was not read" in interpretation
