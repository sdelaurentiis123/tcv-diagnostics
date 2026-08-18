#!/usr/bin/env python3
"""Apply the frozen three-seed B2 gate to verified 85604 evaluation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_acceptance_gate import (  # noqa: E402
    evaluate_b2_architecture_acceptance,
    evaluate_b2_seed_acceptance,
)
from tcv_diagnostics.b2_scoring import (  # noqa: E402
    validate_b2_spectral_materiality,
    validate_b2_transport_event_thresholds,
)
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)


EXPECTED_SEEDS = (1701, 1702, 1703)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--training-matrix-sha256", required=True)
    parser.add_argument("--comparator-matrix", type=Path, required=True)
    parser.add_argument("--comparator-matrix-sha256", required=True)
    parser.add_argument("--event-threshold-result", type=Path, required=True)
    parser.add_argument("--event-threshold-result-sha256", required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--smoke-result-sha256", required=True)
    parser.add_argument(
        "--evaluation-result", type=Path, action="append", required=True
    )
    parser.add_argument("--evaluation-result-sha256", action="append", required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--evaluation-protocol-sha256", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--evaluation-job-id", required=True)
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


def _validate_training_matrix(
    matrix: Mapping[str, Any], *, paper0_commit: str
) -> dict[int, Mapping[str, Any]]:
    if (
        matrix.get("scope") != "phase3_B2_LDM_H2_full_training_matrix_frozen"
        or matrix.get("status") != "completed_pending_bounded_evaluator_smoke"
        or matrix.get("paper0_commit") != paper0_commit
        or matrix.get("development_run") != "85604"
        or matrix.get("held_out_85606_read") is not False
        or matrix.get("seeds") != list(EXPECTED_SEEDS)
        or matrix.get("seed_count") != 3
        or matrix.get("all_training_histories_complete") is not True
        or matrix.get("all_checkpoint_choices_frozen_before_probabilistic_metrics")
        is not True
        or matrix.get("probabilistic_scientific_gate_evaluated") is not False
    ):
        raise ValueError("B2 frozen training matrix contract differs")
    runs = {int(item["seed"]): item for item in matrix.get("runs", [])}
    if tuple(runs) != EXPECTED_SEEDS:
        raise ValueError("B2 frozen training run order differs")
    return runs


def _validate_comparators(
    matrix: Mapping[str, Any],
) -> tuple[dict[int, Mapping[str, Any]], Mapping[str, Any]]:
    if (
        matrix.get("scope") != "phase3_B2_frozen_paired_deterministic_comparators_85604"
        or matrix.get("status") != "completed_before_B2_scientific_acceptance"
        or matrix.get("development_run") != "85604"
        or matrix.get("held_out_85606_read") is not False
        or matrix.get("B2_forecasts_or_scores_read") is not False
        or matrix.get("deterministic_model_retrained") is not False
        or matrix.get("deterministic_checkpoint_reselected") is not False
        or matrix.get("seeds") != list(EXPECTED_SEEDS)
        or matrix.get("seed_count") != 3
        or matrix.get("scientific_acceptance_evaluated") is not False
    ):
        raise ValueError("B2 deterministic comparator matrix contract differs")
    runs = {int(item["seed"]): item for item in matrix.get("runs", [])}
    if tuple(runs) != EXPECTED_SEEDS:
        raise ValueError("B2 deterministic comparator seed order differs")
    best = matrix.get("best_uncompressed", {})
    if best.get("name") != "training_only_toroidal_spectral_AR1":
        raise ValueError("B2 best uncompressed comparator differs")
    return runs, best


def _validate_smoke(
    smoke: Mapping[str, Any],
    *,
    paper0_commit: str,
    training_matrix_sha256: str,
) -> None:
    if (
        smoke.get("scope") != "bounded_non_scientific_B2_evaluator_smoke_85604"
        or smoke.get("status") != "bounded_evaluator_smoke_completed"
        or smoke.get("paper0_commit") != paper0_commit
        or smoke.get("seed") != 1701
        or smoke.get("target_frames") != [498, 502]
        or smoke.get("target_count") != 4
        or smoke.get("ensemble_members") != 32
        or smoke.get("held_out_85606_read") is not False
        or smoke.get("truth_opened_only_after_forecast_hash") is not True
        or smoke.get("full_probabilistic_evaluation_preconditions_passed") is not True
        or smoke.get("probabilistic_scientific_gate_evaluated") is not False
        or smoke.get("training_matrix", {}).get("sha256") != training_matrix_sha256
    ):
        raise ValueError("bounded B2 evaluator smoke contract differs")


def _load_evaluation(
    result_path: Path,
    expected_sha256: str,
    *,
    seed: int,
    paper0_commit: str,
    training_matrix_sha256: str,
    threshold_sha256: str,
    threshold_record: Mapping[str, Any],
    smoke_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    path = verify_input(result_path, expected_sha256)
    result = load_strict_json(path)
    if (
        result.get("scope") != "B2_LDM_H2_full_probabilistic_evaluation_85604"
        or result.get("status") != "completed_pending_frozen_acceptance_gate"
        or result.get("paper0_commit") != paper0_commit
        or result.get("seed") != seed
        or result.get("training_matrix", {}).get("sha256") != training_matrix_sha256
        or result.get("event_threshold_result", {}).get("sha256") != threshold_sha256
        or result.get("bounded_smoke_result", {}).get("sha256") != smoke_sha256
    ):
        raise ValueError(f"B2 evaluation result contract differs for seed {seed}")
    score_record = result.get("score", {})
    score_path = verify_input(Path(score_record["path"]), score_record["sha256"])
    score = load_strict_json(score_path)
    if score.get("model_seed") != seed:
        raise ValueError("B2 evaluation score seed differs")
    if score.get("transport_event_thresholds") != dict(threshold_record):
        raise ValueError("B2 score did not use the frozen threshold/materiality record")
    return result, score


def _family_summary(seed_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        family: {
            "passing_seed_count": sum(
                bool(item["families"][family]["passes"]) for item in seed_records
            ),
            "per_seed": {
                str(item["seed"]): bool(item["families"][family]["passes"])
                for item in seed_records
            },
        }
        for family in ("field", "spectral", "transport")
    }


def main() -> None:
    args = parse_args()
    if len(args.evaluation_result) != 3 or len(args.evaluation_result_sha256) != 3:
        raise ValueError(
            "B2 finalizer requires exactly three evaluation results/hashes"
        )
    for path in (
        args.training_matrix,
        args.comparator_matrix,
        args.event_threshold_result,
        args.smoke_result,
        *args.evaluation_result,
        args.evaluation_manifest,
        args.evaluation_protocol,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    training_path = verify_input(args.training_matrix, args.training_matrix_sha256)
    comparator_path = verify_input(
        args.comparator_matrix, args.comparator_matrix_sha256
    )
    threshold_path = verify_input(
        args.event_threshold_result, args.event_threshold_result_sha256
    )
    smoke_path = verify_input(args.smoke_result, args.smoke_result_sha256)
    manifest_path = verify_input(
        args.evaluation_manifest, args.evaluation_manifest_sha256
    )
    protocol_path = verify_input(
        args.evaluation_protocol, args.evaluation_protocol_sha256
    )
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B2 finalization {output}")
    output.mkdir(parents=True)

    manifest = load_strict_json(manifest_path)
    if (
        manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise ValueError("B2 evaluation manifest scope differs")
    training_runs = _validate_training_matrix(
        load_strict_json(training_path), paper0_commit=args.paper0_commit
    )
    comparator_runs, best_uncompressed = _validate_comparators(
        load_strict_json(comparator_path)
    )
    threshold = load_strict_json(threshold_path)
    validate_b2_transport_event_thresholds(threshold)
    validate_b2_spectral_materiality(threshold.get("spectral_materiality", {}))
    if (
        threshold.get("paper0_commit") != args.paper0_commit
        or threshold.get("held_out_85606_read") is not False
        or threshold.get("inherited_O2_training_materiality", {}).get("sha256")
        != "f76d27df3fdebcd75114401d3747dc6380584ad870cd5caaffa5ca4ce40c662f"
    ):
        raise ValueError("B2 threshold/materiality provenance differs")
    smoke = load_strict_json(smoke_path)
    _validate_smoke(
        smoke,
        paper0_commit=args.paper0_commit,
        training_matrix_sha256=args.training_matrix_sha256,
    )

    seed_records = []
    evaluation_inputs = []
    for seed, result_input, digest in zip(
        EXPECTED_SEEDS, args.evaluation_result, args.evaluation_result_sha256
    ):
        result, score = _load_evaluation(
            result_input,
            digest,
            seed=seed,
            paper0_commit=args.paper0_commit,
            training_matrix_sha256=args.training_matrix_sha256,
            threshold_sha256=args.event_threshold_result_sha256,
            threshold_record=threshold,
            smoke_sha256=args.smoke_result_sha256,
        )
        seed_records.append(
            evaluate_b2_seed_acceptance(
                result=result,
                score=score,
                training_run=training_runs[seed],
                comparator_run=comparator_runs[seed],
                best_uncompressed=best_uncompressed,
                manifest=manifest,
            )
        )
        evaluation_inputs.append(
            {
                "seed": seed,
                "result": {
                    "path": str(Path(result_input).resolve(strict=True)),
                    "sha256": digest,
                },
                "score": dict(result["score"]),
                "forecast": dict(result["forecast"]),
            }
        )
    architecture = evaluate_b2_architecture_acceptance(seed_records)
    passed = architecture["architecture_passes_one_step_B2_gate"]
    final = {
        "schema_version": 1,
        "scope": "phase3_B2_LDM_H2_full_probabilistic_evaluation_matrix_85604",
        "status": (
            "completed_passed_frozen_one_step_gate"
            if passed
            else "completed_failed_frozen_one_step_gate"
        ),
        "scientific_authority": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "paper0_commit": args.paper0_commit,
        "evaluation_job_id": str(args.evaluation_job_id),
        "slurm_job_id": str(args.slurm_job_id),
        "training_matrix": {
            "path": str(training_path),
            "sha256": args.training_matrix_sha256,
        },
        "deterministic_comparator_matrix": {
            "path": str(comparator_path),
            "sha256": args.comparator_matrix_sha256,
        },
        "event_threshold_and_inherited_materiality": {
            "path": str(threshold_path),
            "sha256": args.event_threshold_result_sha256,
        },
        "bounded_evaluator_smoke": {
            "path": str(smoke_path),
            "sha256": args.smoke_result_sha256,
        },
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "evaluation_protocol": {
            "path": str(protocol_path),
            "sha256": args.evaluation_protocol_sha256,
        },
        "evaluation_inputs": evaluation_inputs,
        "family_summary": _family_summary(seed_records),
        "architecture_acceptance": architecture,
        "short_O3_protocol_may_be_frozen": bool(
            architecture["short_O3_protocol_may_be_frozen"]
        ),
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "held_out_85606_access_allowed": False,
        "post_gate_instruction": (
            "freeze_a_separate_short_O3_protocol_before_any_rollout"
            if passed
            else "follow_the_predeclared_failure_branch_without_validation_tuning"
        ),
    }
    result_path = output / "final_matrix.json"
    write_strict_json_atomic(result_path, final)
    index_path = output / "artifact_sha256.txt"
    index_path.write_text(
        f"{sha256_path(result_path)}  {result_path.resolve(strict=True)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": final["status"],
                "architecture_passes": passed,
                "complete_seed_gate_pass_count": architecture[
                    "complete_seed_gate_pass_count"
                ],
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
