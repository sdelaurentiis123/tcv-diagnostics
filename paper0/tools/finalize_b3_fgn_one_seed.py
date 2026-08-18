#!/usr/bin/env python3
"""Apply the frozen B3 seed-1701 scientific gate without rescoring."""

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
from tcv_diagnostics.fgn_acceptance_gate import (  # noqa: E402
    B3_COMPARATOR_SCOPE,
    B3_EVALUATION_SCOPE,
    B3_SCORE_SCOPE,
    B3_TRAINING_SCOPE,
    evaluate_b3_one_seed_acceptance,
)
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)


EXPECTED_MANIFEST_SHA256 = (
    "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--evaluation-result-sha256", required=True)
    parser.add_argument("--comparator-result", type=Path, required=True)
    parser.add_argument("--comparator-result-sha256", required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument("--training-wandb", type=Path, required=True)
    parser.add_argument("--training-wandb-sha256", required=True)
    parser.add_argument("--evaluation-wandb", type=Path, required=True)
    parser.add_argument("--evaluation-wandb-sha256", required=True)
    parser.add_argument("--gate-execution-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != str(expected_commit):
        raise RuntimeError(f"Paper 0 commit {actual} differs from {expected_commit}")
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


def verify_input(path: Path, expected_sha256: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    actual = sha256_path(resolved)
    if actual != str(expected_sha256):
        raise ValueError(f"SHA-256 mismatch for {resolved}: {actual}")
    return resolved


def _verified_reference(
    parent: Mapping[str, Any],
    name: str,
) -> tuple[Path, dict[str, Any]]:
    reference = parent.get(name, {})
    path = verify_input(
        Path(str(reference.get("path", ""))), reference.get("sha256", "")
    )
    return path, load_strict_json(path)


def _family_summary(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "passes": record["passes"],
            "check_count": record["check_count"],
            "failed_check_count": record["failed_check_count"],
            "blocks_passing": record.get("blocks_passing"),
            "blocks_required": record.get("blocks_required"),
        }
        for name, record in acceptance["families"].items()
    }


def main() -> None:
    args = parse_args()
    for path in (
        args.evaluation_result,
        args.comparator_result,
        args.evaluation_manifest,
        args.training_wandb,
        args.evaluation_wandb,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.gate_execution_commit)
    evaluation_path = verify_input(
        args.evaluation_result, args.evaluation_result_sha256
    )
    comparator_path = verify_input(
        args.comparator_result, args.comparator_result_sha256
    )
    manifest_path = verify_input(
        args.evaluation_manifest, args.evaluation_manifest_sha256
    )
    training_wandb_path = verify_input(
        args.training_wandb, args.training_wandb_sha256
    )
    evaluation_wandb_path = verify_input(
        args.evaluation_wandb, args.evaluation_wandb_sha256
    )
    if args.evaluation_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("B3 gate manifest hash differs")
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B3 gate result {output}")
    output.mkdir(parents=True)

    result = load_strict_json(evaluation_path)
    comparator = load_strict_json(comparator_path)
    manifest = load_strict_json(manifest_path)
    training_wandb = load_strict_json(training_wandb_path)
    evaluation_wandb = load_strict_json(evaluation_wandb_path)
    if result.get("scope") != B3_EVALUATION_SCOPE:
        raise ValueError("B3 gate evaluation-result scope differs")
    if comparator.get("scope") != B3_COMPARATOR_SCOPE:
        raise ValueError("B3 gate comparator scope differs")
    if (
        result.get("evaluation_manifest", {}).get("sha256")
        != args.evaluation_manifest_sha256
        or comparator.get("evaluation_manifest", {}).get("sha256")
        != args.evaluation_manifest_sha256
    ):
        raise ValueError("B3 gate input manifest provenance differs")

    training_path, training = _verified_reference(result, "training_result")
    generation_path, generation = _verified_reference(result, "generation")
    score_path, score = _verified_reference(result, "score")
    if training.get("scope") != B3_TRAINING_SCOPE:
        raise ValueError("B3 gate training scope differs")
    if score.get("scope") != B3_SCORE_SCOPE:
        raise ValueError("B3 gate score scope differs")
    if result.get("forecast", {}).get("sha256") != score.get(
        "forecast_artifact", {}
    ).get("sha256"):
        raise ValueError("B3 gate forecast/score identity differs")
    if result.get("training_commit") != training.get("paper0_commit"):
        raise ValueError("B3 gate training commit provenance differs")

    acceptance = evaluate_b3_one_seed_acceptance(
        result=result,
        score=score,
        training=training,
        generation=generation,
        comparator=comparator,
        manifest=manifest,
        training_wandb=training_wandb,
        evaluation_wandb=evaluation_wandb,
    )
    passed = bool(acceptance["passes_complete_one_seed_gate"])
    marginal_joint_failure = bool(
        acceptance["marginal_field_family_passes_but_joint_physics_fails"]
    )
    status = (
        "completed_passed_frozen_one_seed_gate"
        if passed
        else (
            "completed_failed_joint_physics_after_field_family_pass"
            if marginal_joint_failure
            else "completed_failed_frozen_one_seed_gate"
        )
    )
    final = {
        "schema_version": 1,
        "scope": "phase3_B3_FGN_H1_seed1701_scientific_gate_85604",
        "status": status,
        "scientific_authority": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "gate_execution_commit": args.gate_execution_commit,
        "evaluation_commit": result["paper0_commit"],
        "training_commit": result["training_commit"],
        "slurm_job_id": str(args.slurm_job_id),
        "seed": 1701,
        "evaluation_input": {
            "path": str(evaluation_path),
            "sha256": args.evaluation_result_sha256,
        },
        "score_input": {"path": str(score_path), "sha256": result["score"]["sha256"]},
        "generation_input": {
            "path": str(generation_path),
            "sha256": result["generation"]["sha256"],
        },
        "training_input": {
            "path": str(training_path),
            "sha256": result["training_result"]["sha256"],
        },
        "comparator_input": {
            "path": str(comparator_path),
            "sha256": args.comparator_result_sha256,
        },
        "manifest_input": {
            "path": str(manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "training_wandb_input": {
            "path": str(training_wandb_path),
            "sha256": args.training_wandb_sha256,
        },
        "evaluation_wandb_input": {
            "path": str(evaluation_wandb_path),
            "sha256": args.evaluation_wandb_sha256,
        },
        "raw_forecast_changed": False,
        "raw_score_changed": False,
        "metrics_recomputed": False,
        "training_performed": False,
        "inference_performed": False,
        "truth_scoring_performed": False,
        "family_summary": _family_summary(acceptance),
        "acceptance": acceptance,
        "seed1702_1703_replication_protocol_may_be_written": acceptance[
            "seed1702_1703_replication_protocol_may_be_written"
        ],
        "seed1702_1703_training_authorized": False,
        "O3_protocol_may_be_written": False,
        "O3_launch_allowed": False,
        "held_out_85606_access_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "post_gate_instruction": acceptance["disposition"],
    }
    result_path = output / "final_gate.json"
    write_strict_json_atomic(result_path, final)
    index_path = output / "artifact_sha256.txt"
    index_path.write_text(
        f"{sha256_path(result_path)}  {result_path.resolve(strict=True)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "passes_complete_one_seed_gate": passed,
                "family_summary": final["family_summary"],
                "post_gate_instruction": final["post_gate_instruction"],
                "result": str(result_path.resolve(strict=True)),
                "sha256": sha256_path(result_path),
                "held_out_85606_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
