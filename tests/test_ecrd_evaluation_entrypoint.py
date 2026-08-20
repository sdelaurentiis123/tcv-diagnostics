"""Network-free contract tests for the ECRD evaluation entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from paper0.tools.evaluate_ecrd_checkpoint import (
    BASE_PROTOCOL_SHA256,
    HISTORICAL_B5_FORECAST_SHA256,
    audit_full_training_result,
    authorize_evaluation_manifest,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_training import (
    ECRDTrainingConfig,
    frozen_parameter_counts,
    model_config_record,
)


ROOT = Path(__file__).resolve().parents[1]


def test_post_training_manifest_authorizes_only_explicit_arm_seed(tmp_path: Path) -> None:
    hashes = {name: str(index) * 64 for index, name in enumerate(
        (
            "H1_validation_parent",
            "sym_H1_validation_parent",
            "scientific_seed_bank",
            "native_truth_result",
            "geometry_manifest",
            "geometry",
            "event_threshold_result",
        ),
        start=1,
    )}
    manifest = {
        "status": "frozen_after_ECRD_training_before_85604_scientific_evaluation",
        "development_run": "85604",
        "held_out_85606_access_allowed": False,
        "physics_derived_training_loss_allowed": False,
        "base_protocol": {"sha256": BASE_PROTOCOL_SHA256},
        "evaluation_freeze": {
            "sha256": sha256_path(
                ROOT / "paper0/protocol/ECRD_EVALUATION_IMPLEMENTATION_FREEZE.md"
            )
        },
        "authorized_runs": [{"arm": "ECRD", "seed": 1702}],
        "runs": {"ECRD": {"1702": {"kind": "new_full_training"}}},
        "evidence_locks": {
            name: {"sha256": digest} for name, digest in hashes.items()
        },
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = authorize_evaluation_manifest(
        manifest,
        manifest_path=path,
        manifest_sha256=sha256_path(path),
        arm="ECRD",
        seed=1702,
        input_hashes=hashes,
    )
    assert result == {"kind": "new_full_training"}


def test_full_training_audit_requires_complete_data_only_budget(tmp_path: Path) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"immutable checkpoint placeholder")
    checkpoint_sha = sha256_path(checkpoint)
    training_sha = "d" * 64
    commit = "e" * 40
    config = ECRDTrainingConfig(arm="ECRD", seed=1702, mode="full")
    result = {
        "scope": "ECRD_matched_model_development_training_85604",
        "status": "training_completed_checkpoint_selected",
        "mode": "full",
        "arm": "ECRD",
        "seed": 1702,
        "development_run": "85604",
        "training": json.loads(json.dumps(config.to_record())),
        "model": model_config_record("ECRD"),
        "parameter_count": frozen_parameter_counts()["ECRD"],
        "completed_epochs": 100,
        "completed_optimizer_steps": 10_800,
        "target_presentations": 43_000,
        "candidate_count": 20,
        "checkpoint_reload_bitwise_exact": True,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "paper0_commit": commit,
        "artifacts": {
            "selected_checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
            }
        },
    }
    audited = audit_full_training_result(
        result,
        arm="ECRD",
        seed=1702,
        expected_sha256=training_sha,
        manifest_run={
            "kind": "new_full_training",
            "training_result_sha256": training_sha,
            "training_commit": commit,
            "selected_checkpoint_sha256": checkpoint_sha,
        },
    )
    assert audited["checkpoint_sha256"] == checkpoint_sha
    assert audited["training_commit"] == commit


def test_historical_b5_hash_is_inherited_not_reinvented() -> None:
    source = json.loads(
        (ROOT / "paper0/manifests/phase3_b5_covariance_localization_85604.json").read_text()
    )
    assert source["evidence_locks"]["B5_forecast"]["sha256"] == (
        HISTORICAL_B5_FORECAST_SHA256
    )


def test_entrypoint_keeps_scientific_scores_local() -> None:
    source = (ROOT / "paper0/tools/evaluate_ecrd_checkpoint.py").read_text()
    assert "wandb" not in source.lower()
    assert "score_ecrd_forecast" in source
    assert "truth_opened_only_after_forecast_hash" in source
