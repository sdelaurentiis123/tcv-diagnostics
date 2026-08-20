#!/usr/bin/env python3
"""Freeze the post-training, pre-forecast ECRD evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.ecrd_training import (  # noqa: E402
    ECRD_ARMS,
    ECRD_MODEL_SEEDS,
)
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)


BASE_PROTOCOL = ROOT / "paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
EVALUATION_FREEZE = (
    ROOT / "paper0/protocol/ECRD_EVALUATION_IMPLEMENTATION_FREEZE.md"
)
TRAINING_COMMIT = "d822ee2147a98713f1b2ecdfd0f5a4077eded062"
TRAINING_ARRAY_JOB_ID = "6913340"
HISTORICAL_B5_FORECAST_SHA256 = (
    "1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b"
)

NEW_RUNS = (
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
)

CODE_LOCKS = (
    "paper0/tools/evaluate_ecrd_checkpoint.py",
    "paper0/tools/summarize_ecrd_model_ladder.py",
    "src/tcv_diagnostics/ecrd_forecast.py",
    "src/tcv_diagnostics/ecrd_scoring.py",
    "src/tcv_diagnostics/ecrd_acceptance.py",
    "src/tcv_diagnostics/ecrd_training.py",
    "src/tcv_diagnostics/models/ecrd.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-finalization-result", type=Path, required=True)
    parser.add_argument("--training-finalization-result-sha256", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--h1-validation-parent", type=Path, required=True)
    parser.add_argument("--h1-validation-parent-sha256", required=True)
    parser.add_argument("--sym-h1-validation-parent", type=Path, required=True)
    parser.add_argument("--sym-h1-validation-parent-sha256", required=True)
    parser.add_argument("--scientific-seed-bank", type=Path, required=True)
    parser.add_argument("--scientific-seed-bank-sha256", required=True)
    parser.add_argument("--historical-b5-forecast", type=Path, required=True)
    parser.add_argument("--historical-b5-forecast-sha256", required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--event-threshold-result", type=Path, required=True)
    parser.add_argument("--event-threshold-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != str(expected_commit):
        raise RuntimeError(f"Paper 0 commit mismatch: {actual} != {expected_commit}")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    source = Path(path).resolve(strict=True)
    assert_development_path(source)
    observed = sha256_path(source)
    if observed != str(expected_sha256):
        raise RuntimeError(f"{label} SHA-256 differs: {observed}")
    return source


def audit_training_finalization(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if (
        result.get("scope") != "ECRD_matched_full_training_finalization_85604"
        or result.get("status") != "all_eleven_new_training_tasks_verified"
        or result.get("training_commit") != TRAINING_COMMIT
        or result.get("array_job_id") != TRAINING_ARRAY_JOB_ID
        or result.get("development_run") != "85604"
        or result.get("total_ladder_runs") != 12
        or result.get("all_training_artifact_indices_verified") is not True
        or result.get("all_wandb_runs_finished") is not True
        or result.get("paired_training_order_verified") is not True
        or result.get("paired_validation_seed_bank_verified") is not True
        or result.get("scientific_result") is not False
        or result.get("physics_metric_evaluated") is not False
        or result.get("guard_frames_read") is not False
        or result.get("held_out_85606_read") is not False
        or result.get("scientific_forecast_generated") is not False
    ):
        raise RuntimeError("ECRD training-finalization contract differs")
    runs = list(result.get("new_runs", ()))
    observed = [
        (run.get("array_index"), run.get("arm"), run.get("seed")) for run in runs
    ]
    if observed != list(NEW_RUNS):
        raise RuntimeError("ECRD finalized training matrix differs")
    return runs


def _new_run_locks(runs: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {arm: {} for arm in ECRD_ARMS}
    for run in runs:
        arm = str(run["arm"])
        seed = int(run["seed"])
        result_record = run["result"]
        selected = run["selected_checkpoint"]
        result_path = verify_input(
            Path(result_record["path"]),
            str(result_record["sha256"]),
            f"{arm} seed {seed} training result",
        )
        checkpoint_path = verify_input(
            Path(selected["path"]),
            str(selected["sha256"]),
            f"{arm} seed {seed} selected checkpoint",
        )
        records[arm][str(seed)] = {
            "kind": "new_full_training",
            "training_result_path": str(result_path),
            "training_result_sha256": str(result_record["sha256"]),
            "training_commit": TRAINING_COMMIT,
            "selected_checkpoint_path": str(checkpoint_path),
            "selected_checkpoint_sha256": str(selected["sha256"]),
            "selected_completed_epoch": int(selected["completed_epoch"]),
            "selection_metric": "data_only_three_block_validation_objective",
            "physics_metric_used_for_selection": False,
        }
    return records


def _historical_run_lock(
    training_manifest: Mapping[str, Any],
    *,
    forecast_path: Path,
    forecast_sha256: str,
) -> dict[str, Any]:
    historical = training_manifest.get("historical_B5_seed1701", {})
    if (
        historical.get("reuse_authorized") is not True
        or historical.get("artifact_index_verified") is not True
        or historical.get("held_out_85606_read") is not False
    ):
        raise RuntimeError("historical B5 training reuse contract differs")
    verify_input(
        Path(historical["result_path"]),
        str(historical["result_sha256"]),
        "historical B5 training result",
    )
    verify_input(
        Path(historical["selected_checkpoint_path"]),
        str(historical["selected_checkpoint_sha256"]),
        "historical B5 selected checkpoint",
    )
    return {
        "kind": "historical_B5_seed1701_forecast_reuse",
        "training_result_path": str(
            Path(historical["result_path"]).resolve(strict=True)
        ),
        "training_result_sha256": str(historical["result_sha256"]),
        "training_commit": str(historical["training_commit"]),
        "selected_checkpoint_path": str(
            Path(historical["selected_checkpoint_path"]).resolve(strict=True)
        ),
        "selected_checkpoint_sha256": str(
            historical["selected_checkpoint_sha256"]
        ),
        "forecast_path": str(forecast_path),
        "forecast_sha256": forecast_sha256,
        "forecast_reused_without_mutation": True,
    }


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    runtime_paths = (
        args.training_finalization_result,
        args.training_manifest,
        args.artifact_root,
        args.h1_validation_parent,
        args.sym_h1_validation_parent,
        args.scientific_seed_bank,
        args.historical_b5_forecast,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.event_threshold_result,
        args.output,
    )
    if any("85606" in str(path).lower() for path in runtime_paths):
        raise ValueError("held-out paths are prohibited during ECRD evaluation freeze")
    output = Path(args.output)
    assert_development_path(output)
    if output.exists():
        raise FileExistsError(output)

    finalization_path = verify_input(
        args.training_finalization_result,
        args.training_finalization_result_sha256,
        "ECRD training finalization",
    )
    finalization = load_strict_json(finalization_path)
    runs = audit_training_finalization(finalization)
    training_manifest_path = verify_input(
        args.training_manifest,
        args.training_manifest_sha256,
        "ECRD full-training manifest",
    )
    training_manifest = load_strict_json(training_manifest_path)
    if (
        training_manifest.get("status")
        != "frozen_after_passing_ECRD_smoke_before_full_training"
        or training_manifest.get("development_run") != "85604"
        or training_manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("ECRD full-training manifest scope differs")

    artifact_root = Path(args.artifact_root).resolve(strict=True)
    assert_development_path(artifact_root)
    model_lock = training_manifest["evidence_locks"]["model_dataset"]
    model_files = {
        "manifest_sha256": verify_input(
            artifact_root / "model_dataset_manifest.json",
            model_lock["manifest_sha256"],
            "model-data manifest",
        ),
        "normalization_sha256": verify_input(
            artifact_root / "normalization.json",
            model_lock["normalization_sha256"],
            "model-data normalization",
        ),
        "artifact_index_sha256": verify_input(
            artifact_root / "artifact_sha256.txt",
            model_lock["artifact_index_sha256"],
            "model-data artifact index",
        ),
    }
    inputs = {
        "H1_validation_parent": (
            verify_input(
                args.h1_validation_parent,
                args.h1_validation_parent_sha256,
                "H1 validation parent",
            ),
            args.h1_validation_parent_sha256,
        ),
        "sym_H1_validation_parent": (
            verify_input(
                args.sym_h1_validation_parent,
                args.sym_h1_validation_parent_sha256,
                "symmetrized H1 validation parent",
            ),
            args.sym_h1_validation_parent_sha256,
        ),
        "scientific_seed_bank": (
            verify_input(
                args.scientific_seed_bank,
                args.scientific_seed_bank_sha256,
                "scientific sampler seed bank",
            ),
            args.scientific_seed_bank_sha256,
        ),
        "native_truth_result": (
            verify_input(
                args.native_truth_result,
                args.native_truth_result_sha256,
                "native truth result",
            ),
            args.native_truth_result_sha256,
        ),
        "geometry_manifest": (
            verify_input(
                args.geometry_manifest,
                args.geometry_manifest_sha256,
                "geometry manifest",
            ),
            args.geometry_manifest_sha256,
        ),
        "geometry": (
            verify_input(args.geometry, args.geometry_sha256, "native geometry"),
            args.geometry_sha256,
        ),
        "event_threshold_result": (
            verify_input(
                args.event_threshold_result,
                args.event_threshold_result_sha256,
                "event-threshold result",
            ),
            args.event_threshold_result_sha256,
        ),
    }
    if args.historical_b5_forecast_sha256 != HISTORICAL_B5_FORECAST_SHA256:
        raise RuntimeError("historical B5 forecast identity differs")
    historical_forecast = verify_input(
        args.historical_b5_forecast,
        args.historical_b5_forecast_sha256,
        "historical B5 M32 forecast",
    )

    run_locks = _new_run_locks(runs)
    run_locks["B5"]["1701"] = _historical_run_lock(
        training_manifest,
        forecast_path=historical_forecast,
        forecast_sha256=args.historical_b5_forecast_sha256,
    )
    if any(tuple(sorted(map(int, run_locks[arm]))) != ECRD_MODEL_SEEDS for arm in ECRD_ARMS):
        raise RuntimeError("ECRD evaluation run matrix is incomplete")

    evidence_locks = {
        name: {"path": str(path), "sha256": digest}
        for name, (path, digest) in inputs.items()
    }
    evidence_locks["model_dataset"] = {
        "root": str(artifact_root),
        **{name: model_lock[name] for name in model_files},
    }
    evidence_locks["training_finalization"] = {
        "path": str(finalization_path),
        "sha256": args.training_finalization_result_sha256,
    }
    evidence_locks["historical_B5_forecast"] = {
        "path": str(historical_forecast),
        "sha256": args.historical_b5_forecast_sha256,
    }
    evidence_locks["evaluation_code"] = {
        relative: {"sha256": sha256_path(ROOT / relative)}
        for relative in CODE_LOCKS
    }
    manifest = {
        "schema_version": 1,
        "status": "frozen_after_ECRD_training_before_85604_scientific_evaluation",
        "freeze_date": "2026-08-20",
        "development_run": "85604",
        "held_out_85606_access_allowed": False,
        "physics_derived_training_loss_allowed": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "base_protocol": {
            "path": str(BASE_PROTOCOL.relative_to(ROOT)),
            "sha256": sha256_path(BASE_PROTOCOL),
        },
        "evaluation_freeze": {
            "path": str(EVALUATION_FREEZE.relative_to(ROOT)),
            "sha256": sha256_path(EVALUATION_FREEZE),
        },
        "training_manifest": {
            "path": str(training_manifest_path),
            "sha256": args.training_manifest_sha256,
        },
        "authorized_runs": [
            {"arm": arm, "seed": seed}
            for arm in ECRD_ARMS
            for seed in ECRD_MODEL_SEEDS
        ],
        "runs": run_locks,
        "evidence_locks": evidence_locks,
        "evaluation": {
            "target_frames": [498, 624],
            "validation_blocks": {
                "V00": [498, 540],
                "V01": [540, 582],
                "V02": [582, 624],
            },
            "ensemble_members": 32,
            "EDM_steps": 18,
            "network_evaluations_per_member": 35,
            "paired_member_seeds_across_arms": True,
            "posthoc_inflation_allowed": False,
            "target_truth_opened_only_after_forecast_hash": True,
            "acceptance_rule": "seven_family_ECRD_gate",
        },
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "assimilation_authorized": False,
        "diagnostic_ranking_authorized": False,
        "steering_authorized": False,
    }
    output.mkdir(parents=True)
    manifest_path = output / "ecrd_scientific_evaluation_85604.json"
    write_strict_json_atomic(manifest_path, manifest)
    result = {
        "schema_version": 1,
        "scope": "post_training_pre_forecast_ECRD_evaluation_freeze_85604",
        "status": "evaluation_manifest_frozen_without_scientific_forecast",
        "development_run": "85604",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorized_run_count": 12,
        "manifest": {
            "path": str(manifest_path.resolve(strict=True)),
            "sha256": sha256_path(manifest_path),
        },
        "training_finalization": {
            "path": str(finalization_path),
            "sha256": args.training_finalization_result_sha256,
        },
        "physics_metric_inspected": False,
        "scientific_forecast_generated": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    index_path = output / "artifact_sha256.txt"
    index_path.write_text(
        "\n".join(
            f"{sha256_path(path)}  {path.resolve(strict=True)}"
            for path in (manifest_path, result_path)
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest": result["manifest"],
                "result_sha256": sha256_path(result_path),
                "artifact_index_sha256": sha256_path(index_path),
                "held_out_85606_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
