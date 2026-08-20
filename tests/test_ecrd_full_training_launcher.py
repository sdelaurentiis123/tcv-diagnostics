"""Frozen-manifest and launcher tests for matched ECRD full training."""

from __future__ import annotations

import json
from pathlib import Path

from paper0.tools.train_ecrd import authorize_manifest
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_training import ECRD_ARMS, ECRD_MODEL_SEEDS


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/ecrd_full_training_85604.json"
LAUNCHER = ROOT / "cluster/ecrd_full_training.sbatch"


def test_full_training_manifest_authorizes_exact_four_by_three_ladder() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == (
        "frozen_after_passing_ECRD_smoke_before_full_training"
    )
    assert tuple(manifest["authorized_arms"]) == ECRD_ARMS
    assert tuple(manifest["authorized_seeds"]) == ECRD_MODEL_SEEDS
    assert manifest["full_training_authorized"] is True
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["symmetrized_parent_use"] == {
        "artifact_authority": "scientific_H100_parent",
        "execution_device": "h100",
        "authorized_modes": ["smoke", "full"],
    }
    locks = manifest["evidence_locks"]
    input_hashes = {
        name: locks[name]["sha256"]
        for name in (
            "H1_training_parent",
            "H1_validation_parent",
            "sym_H1_training_parent",
            "sym_H1_validation_parent",
        )
    }
    digest = sha256_path(MANIFEST)
    for arm in ECRD_ARMS:
        for seed in ECRD_MODEL_SEEDS:
            record = authorize_manifest(
                manifest,
                manifest_path=MANIFEST,
                manifest_sha256=digest,
                mode="full",
                arm=arm,
                seed=seed,
                input_hashes=input_hashes,
            )
            assert record["authorized"] is True
            assert record["held_out_85606_read"] is False


def test_new_training_matrix_excludes_only_historical_b5_seed1701() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    matrix = [
        (item["array_index"], item["arm"], item["seed"])
        for item in manifest["new_training_matrix"]
    ]
    expected = [
        (0, "B5", 1702),
        (1, "B5", 1703),
        (2, "B5-Context", 1701),
        (3, "B5-Context", 1702),
        (4, "B5-Context", 1703),
        (5, "ECRD", 1701),
        (6, "ECRD", 1702),
        (7, "ECRD", 1703),
        (8, "ECRD-History", 1701),
        (9, "ECRD-History", 1702),
        (10, "ECRD-History", 1703),
    ]
    assert matrix == expected
    historical = manifest["historical_B5_seed1701"]
    assert historical["reuse_authorized"] is True
    assert historical["artifact_index_verified"] is True
    assert historical["completed_optimizer_steps"] == 10800
    assert historical["held_out_85606_read"] is False


def test_full_training_launcher_is_h100_hash_locked_and_truth_safe() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-10%2" in source
    assert "#SBATCH --constraint=h100" in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert sha256_path(MANIFEST) in source
    assert "--mode full" in source
    assert "--wandb-group \"${WANDB_GROUP}\"" in source
    assert '"completed_optimizer_steps": 10800' in source
    assert '"candidate_count": 20' in source
    assert '"physics_derived_loss_used": False' in source
    assert '"held_out_85606_read": False' in source
    assert '"scientific_forecast_generated": False' in source
    assert "evaluate_ecrd_checkpoint.py" not in source
    assert "run_assimilation" not in source
    assert '"assimilation_performed": False' in source


def test_frozen_result_records_support_full_training_boundary() -> None:
    smoke = json.loads(
        (ROOT / "paper0/results/ecrd_engineering_smoke_6912495.json").read_text()
    )
    parent = json.loads(
        (ROOT / "paper0/results/ecrd_sym_h1_parent_h100_6913234.json").read_text()
    )
    comparison = json.loads(
        (
            ROOT
            / "paper0/results/ecrd_parent_execution_comparison_6913271.json"
        ).read_text()
    )
    assert smoke["status"] == "all_four_arms_passed_and_independently_verified"
    assert smoke["scientific_result"] is False
    assert parent["artifact_authority"] == "scientific_H100_parent"
    assert parent["full_training_authorized"] is True
    assert comparison["CPU_parent_promoted"] is False
    assert comparison["full_training_parent_authority"] == (
        "scientific_H100_parent_only"
    )
    assert comparison["comparison"]["train"][
        "engineering_consistency_guard_passed"
    ] is True
    assert comparison["comparison"]["validation"][
        "engineering_consistency_guard_passed"
    ] is True
