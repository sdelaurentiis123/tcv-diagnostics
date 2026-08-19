#!/usr/bin/env python3
"""Apply the frozen B4 H-det/H-prob gate without inference or rescoring."""

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
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.pde_refiner_acceptance_gate import (  # noqa: E402
    B4_COMPARATOR_SCOPE,
    B4_EVALUATION_SCOPE,
    B4_FINAL_SCORE_SCOPE,
    B4_GENERATION_SCOPE,
    B4_MANIFEST_SHA256,
    B4_STAGE_SCORE_SCOPE,
    B4_TRAINING_SCOPE,
    evaluate_b4_one_seed_acceptance,
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
    parent: Mapping[str, Any], name: str
) -> tuple[Path, dict[str, Any]]:
    reference = parent.get(name, {})
    path = verify_input(
        Path(str(reference.get("path", ""))), str(reference.get("sha256", ""))
    )
    return path, load_strict_json(path)


def _family_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "passes": family["passes"],
            "passes_overall": family["passes_overall"],
            "blocks_passing": family["blocks_passing"],
            "blocks_required": family["blocks_required"],
            "check_count": family["check_count"],
            "failed_check_count": family["failed_check_count"],
        }
        for name, family in record["families"].items()
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
    if args.evaluation_manifest_sha256 != B4_MANIFEST_SHA256:
        raise RuntimeError("B4 gate manifest hash differs")
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B4 gate result {output}")
    output.mkdir(parents=True)

    result = load_strict_json(evaluation_path)
    comparator = load_strict_json(comparator_path)
    manifest = load_strict_json(manifest_path)
    training_wandb = load_strict_json(training_wandb_path)
    evaluation_wandb = load_strict_json(evaluation_wandb_path)
    if result.get("scope") != B4_EVALUATION_SCOPE:
        raise ValueError("B4 gate evaluation-result scope differs")
    if comparator.get("scope") != B4_COMPARATOR_SCOPE:
        raise ValueError("B4 gate comparator scope differs")
    if result.get("evaluation_manifest", {}).get("sha256") != B4_MANIFEST_SHA256:
        raise ValueError("B4 gate evaluation manifest provenance differs")

    training_path, training = _verified_reference(result, "training_result")
    generation_path, generation = _verified_reference(result, "generation")
    score_path, score = _verified_reference(result, "final_score")
    stage_score_path, stage_score = _verified_reference(result, "stage_score")
    if training.get("scope") != B4_TRAINING_SCOPE:
        raise ValueError("B4 gate training scope differs")
    if generation.get("scope") != B4_GENERATION_SCOPE:
        raise ValueError("B4 gate generation scope differs")
    if score.get("scope") != B4_FINAL_SCORE_SCOPE:
        raise ValueError("B4 gate final-score scope differs")
    if stage_score.get("scope") != B4_STAGE_SCORE_SCOPE:
        raise ValueError("B4 gate stage-score scope differs")
    if result.get("final_forecast", {}).get("sha256") != score.get(
        "forecast_artifact", {}
    ).get("sha256"):
        raise ValueError("B4 gate final forecast/score identity differs")
    if result.get("stage_forecast", {}).get("sha256") != stage_score.get(
        "stage_artifact", {}
    ).get("sha256"):
        raise ValueError("B4 gate stage forecast/score identity differs")
    if result.get("training_commit") != training.get("paper0_commit"):
        raise ValueError("B4 gate training commit provenance differs")

    acceptance = evaluate_b4_one_seed_acceptance(
        result=result,
        score=score,
        stage_score=stage_score,
        training=training,
        generation=generation,
        comparator=comparator,
        manifest=manifest,
        training_wandb=training_wandb,
        evaluation_wandb=evaluation_wandb,
    )
    h_det_pass = bool(acceptance["H_det"]["passes"])
    h_prob_pass = bool(acceptance["H_prob"]["passes"])
    if h_det_pass and h_prob_pass:
        status = "completed_joint_H_det_H_prob_pass"
    elif h_det_pass:
        status = "completed_H_det_pass_H_prob_fail"
    elif h_prob_pass:
        status = "completed_H_prob_pass_H_det_fail"
    else:
        status = "completed_joint_H_det_H_prob_fail"
    final = {
        "schema_version": 1,
        "scope": "phase3_B4_PDE_Refiner_H1_seed1701_scientific_gate_85604",
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
        "inputs": {
            "evaluation": {
                "path": str(evaluation_path),
                "sha256": args.evaluation_result_sha256,
            },
            "training": {
                "path": str(training_path),
                "sha256": result["training_result"]["sha256"],
            },
            "generation": {
                "path": str(generation_path),
                "sha256": result["generation"]["sha256"],
            },
            "final_score": {
                "path": str(score_path),
                "sha256": result["final_score"]["sha256"],
            },
            "stage_score": {
                "path": str(stage_score_path),
                "sha256": result["stage_score"]["sha256"],
            },
            "comparator": {
                "path": str(comparator_path),
                "sha256": args.comparator_result_sha256,
            },
            "manifest": {
                "path": str(manifest_path),
                "sha256": args.evaluation_manifest_sha256,
            },
            "training_wandb": {
                "path": str(training_wandb_path),
                "sha256": args.training_wandb_sha256,
            },
            "evaluation_wandb": {
                "path": str(evaluation_wandb_path),
                "sha256": args.evaluation_wandb_sha256,
            },
        },
        "raw_forecasts_changed": False,
        "raw_scores_changed": False,
        "metrics_recomputed": False,
        "training_performed": False,
        "inference_performed": False,
        "truth_scoring_performed": False,
        "summary": {
            "H_det": {
                "passes": h_det_pass,
                "joint_blocks_passing": acceptance["H_det"][
                    "joint_blocks_passing"
                ],
                "families": _family_summary(acceptance["H_det"]),
            },
            "H_prob": {
                "passes": h_prob_pass,
                "joint_blocks_passing": acceptance["H_prob"][
                    "joint_blocks_passing"
                ],
                "families": _family_summary(acceptance["H_prob"]),
            },
            "stagewise_repair_passes": stage_score["stagewise_repair"]["passes"],
        },
        "acceptance": acceptance,
        "seed1702_1703_replication_protocol_may_be_written": acceptance[
            "seed1702_1703_replication_protocol_may_be_written"
        ],
        "seed1702_1703_training_authorized": False,
        "O3_protocol_may_be_written": acceptance["O3_protocol_may_be_written"],
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
                "H_det_passes": h_det_pass,
                "H_prob_passes": h_prob_pass,
                "joint_pass": h_det_pass and h_prob_pass,
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
