#!/usr/bin/env python3
"""Apply A016 to the three immutable B2 score records without rescoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper0.tools.finalize_b2_evaluation import (  # noqa: E402
    EXPECTED_SEEDS,
    _family_summary,
    _load_evaluation,
    _validate_comparators,
    _validate_smoke,
    _validate_training_matrix,
    verify_checkout,
    verify_input,
)
from tcv_diagnostics.b2_acceptance_gate_event_eligibility import (  # noqa: E402
    EVENT_BLOCK_POLICY,
    evaluate_b2_architecture_acceptance_event_eligible,
    evaluate_b2_seed_acceptance_event_eligible,
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


ORIGINAL_MATRIX_SHA256 = (
    "cd5d3a22b1a5f665c493417c3ea47bc7fd21d731e116f35a6a84eae68b462fd6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-matrix", type=Path, required=True)
    parser.add_argument("--original-matrix-sha256", required=True)
    parser.add_argument("--amendment-manifest", type=Path, required=True)
    parser.add_argument("--amendment-manifest-sha256", required=True)
    parser.add_argument("--amendment-protocol", type=Path, required=True)
    parser.add_argument("--amendment-protocol-sha256", required=True)
    parser.add_argument("--gate-execution-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _validate_original_matrix(matrix: Mapping[str, Any], *, digest: str) -> None:
    if digest != ORIGINAL_MATRIX_SHA256:
        raise ValueError("A016 original B2 matrix hash differs")
    if (
        matrix.get("scope")
        != "phase3_B2_LDM_H2_full_probabilistic_evaluation_matrix_85604"
        or matrix.get("status") != "completed_failed_frozen_one_step_gate"
        or matrix.get("scientific_authority") is not True
        or matrix.get("development_run") != "85604"
        or matrix.get("held_out_85606_read") is not False
        or matrix.get("slurm_job_id") != "6897564"
        or matrix.get("paper0_commit")
        != "361f0f27a9ece3b56f529a72c2fcfa19aa0be719"
        or matrix.get("architecture_acceptance", {}).get(
            "architecture_passes_one_step_B2_gate"
        )
        is not False
        or matrix.get("O3_launch_allowed") is not False
        or matrix.get("assimilation_allowed") is not False
        or matrix.get("diagnostic_ranking_allowed") is not False
    ):
        raise ValueError("A016 original B2 matrix contract differs")
    inputs = matrix.get("evaluation_inputs", [])
    if tuple(int(item.get("seed", -1)) for item in inputs) != EXPECTED_SEEDS:
        raise ValueError("A016 original B2 evaluation-input order differs")


def _validate_amendment(
    manifest: Mapping[str, Any],
    *,
    original_digest: str,
    protocol_digest: str,
) -> None:
    if (
        manifest.get("scope")
        != "phase3_B2_truth_event_eligibility_amendment_85604"
        or manifest.get("status")
        != "frozen_before_amended_evaluator_implementation_or_execution"
        or manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("outcome_informed_amendment") is not True
        or manifest.get("original_result", {}).get("sha256") != original_digest
        or manifest.get("original_result", {}).get("retained_immutable") is not True
        or manifest.get("protocol", {}).get("sha256") != protocol_digest
        or manifest.get("consistent_rerun", {}).get("seeds")
        != list(EXPECTED_SEEDS)
        or manifest.get("consistent_rerun", {}).get("gate_only_reduction_allowed")
        is not True
        or manifest.get("consistent_rerun", {}).get("inference_allowed") is not False
        or manifest.get("consistent_rerun", {}).get("truth_scoring_allowed")
        is not False
    ):
        raise ValueError("A016 amendment manifest contract differs")


def main() -> None:
    args = parse_args()
    for path in (
        args.original_matrix,
        args.amendment_manifest,
        args.amendment_protocol,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.gate_execution_commit)
    original_path = verify_input(
        args.original_matrix, args.original_matrix_sha256
    )
    amendment_manifest_path = verify_input(
        args.amendment_manifest, args.amendment_manifest_sha256
    )
    amendment_protocol_path = verify_input(
        args.amendment_protocol, args.amendment_protocol_sha256
    )
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite A016 matrix {output}")
    output.mkdir(parents=True)

    original = load_strict_json(original_path)
    _validate_original_matrix(original, digest=args.original_matrix_sha256)
    amendment_manifest = load_strict_json(amendment_manifest_path)
    _validate_amendment(
        amendment_manifest,
        original_digest=args.original_matrix_sha256,
        protocol_digest=args.amendment_protocol_sha256,
    )

    original_commit = str(original["paper0_commit"])
    training_record = original["training_matrix"]
    comparator_record = original["deterministic_comparator_matrix"]
    threshold_record = original["event_threshold_and_inherited_materiality"]
    smoke_record = original["bounded_evaluator_smoke"]
    manifest_record = original["evaluation_manifest"]

    training_path = verify_input(
        Path(training_record["path"]), training_record["sha256"]
    )
    comparator_path = verify_input(
        Path(comparator_record["path"]), comparator_record["sha256"]
    )
    threshold_path = verify_input(
        Path(threshold_record["path"]), threshold_record["sha256"]
    )
    smoke_path = verify_input(Path(smoke_record["path"]), smoke_record["sha256"])
    evaluation_manifest_path = verify_input(
        Path(manifest_record["path"]), manifest_record["sha256"]
    )
    training_runs = _validate_training_matrix(
        load_strict_json(training_path), paper0_commit=original_commit
    )
    comparator_runs, best_uncompressed = _validate_comparators(
        load_strict_json(comparator_path)
    )
    threshold = load_strict_json(threshold_path)
    validate_b2_transport_event_thresholds(threshold)
    validate_b2_spectral_materiality(threshold["spectral_materiality"])
    _validate_smoke(
        load_strict_json(smoke_path),
        paper0_commit=original_commit,
        training_matrix_sha256=training_record["sha256"],
    )
    evaluation_manifest = load_strict_json(evaluation_manifest_path)

    seed_records = []
    input_records = []
    for seed, item in zip(EXPECTED_SEEDS, original["evaluation_inputs"]):
        result_record = item["result"]
        result, score = _load_evaluation(
            Path(result_record["path"]),
            result_record["sha256"],
            seed=seed,
            paper0_commit=original_commit,
            training_matrix_sha256=training_record["sha256"],
            threshold_sha256=threshold_record["sha256"],
            threshold_record=threshold,
            smoke_sha256=smoke_record["sha256"],
        )
        if dict(result["score"]) != dict(item["score"]):
            raise ValueError(f"A016 score identity differs for seed {seed}")
        seed_records.append(
            evaluate_b2_seed_acceptance_event_eligible(
                result=result,
                score=score,
                training_run=training_runs[seed],
                comparator_run=comparator_runs[seed],
                best_uncompressed=best_uncompressed,
                manifest=evaluation_manifest,
            )
        )
        input_records.append(
            {
                "seed": seed,
                "result": dict(result_record),
                "score": dict(item["score"]),
                "forecast": dict(item["forecast"]),
            }
        )

    architecture = evaluate_b2_architecture_acceptance_event_eligible(seed_records)
    amended_pass = bool(architecture["architecture_passes_one_step_B2_gate"])
    original_pass = bool(
        original["architecture_acceptance"]["architecture_passes_one_step_B2_gate"]
    )
    final = {
        "schema_version": 2,
        "scope": "phase3_B2_LDM_H2_A016_event_eligible_gate_matrix_85604",
        "status": (
            "completed_passed_amended_one_step_gate"
            if amended_pass
            else "completed_failed_amended_one_step_gate"
        ),
        "scientific_authority": True,
        "post_result_amendment": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "gate_execution_commit": args.gate_execution_commit,
        "original_forecast_evaluation_commit": original_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "event_block_policy": EVENT_BLOCK_POLICY,
        "original_matrix": {
            "path": str(original_path),
            "sha256": args.original_matrix_sha256,
            "status": original["status"],
            "retained_immutable": True,
        },
        "amendment_manifest": {
            "path": str(amendment_manifest_path),
            "sha256": args.amendment_manifest_sha256,
        },
        "amendment_protocol": {
            "path": str(amendment_protocol_path),
            "sha256": args.amendment_protocol_sha256,
        },
        "evaluation_inputs": input_records,
        "raw_forecasts_changed": False,
        "raw_scores_changed": False,
        "metrics_recomputed": False,
        "training_performed": False,
        "inference_performed": False,
        "family_summary": _family_summary(seed_records),
        "architecture_acceptance": architecture,
        "architecture_decision_changed": amended_pass != original_pass,
        "original_architecture_passed": original_pass,
        "amended_architecture_passed": amended_pass,
        "short_O3_protocol_may_be_frozen": amended_pass,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "held_out_85606_access_allowed": False,
        "post_gate_instruction": (
            "freeze_a_separate_short_O3_protocol_before_any_rollout"
            if amended_pass
            else "continue_to_the_predeclared_FGN_or_joint_residual_failure_branch"
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
                "original_architecture_passed": original_pass,
                "amended_architecture_passed": amended_pass,
                "architecture_decision_changed": amended_pass != original_pass,
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
