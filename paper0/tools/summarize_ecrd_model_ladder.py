#!/usr/bin/env python3
"""Apply the frozen three-seed ECRD gate to twelve stored 85604 scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_scoring import (  # noqa: E402
    validate_b2_spectral_materiality,
)
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.ecrd_acceptance import (  # noqa: E402
    evaluate_ecrd_model_ladder,
)
from tcv_diagnostics.ecrd_training import ECRD_ARMS, ECRD_MODEL_SEEDS  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        metavar="ARM:SEED:PATH:SHA256",
        help="repeat exactly once for every four-arm/three-seed score",
    )
    parser.add_argument("--event-threshold-result", type=Path, required=True)
    parser.add_argument("--event-threshold-result-sha256", required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
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
        raise RuntimeError("ECRD reducer checkout differs")
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
        raise RuntimeError(f"ECRD reducer checkout is dirty:\n{dirty}")


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    observed = sha256_path(resolved)
    if observed != str(expected_sha256):
        raise RuntimeError(f"{label} SHA-256 differs: {observed}")
    return resolved


def parse_score_specifications(
    specifications: list[str],
) -> dict[str, dict[int, tuple[Path, str]]]:
    matrix: dict[str, dict[int, tuple[Path, str]]] = {
        arm: {} for arm in ECRD_ARMS
    }
    for specification in specifications:
        parts = str(specification).split(":", 3)
        if len(parts) != 4:
            raise ValueError("ECRD score specification must have four fields")
        arm, seed_text, path_text, digest = parts
        if arm not in ECRD_ARMS:
            raise ValueError(f"unknown ECRD score arm {arm!r}")
        seed = int(seed_text)
        if seed not in ECRD_MODEL_SEEDS or seed in matrix[arm]:
            raise ValueError("ECRD score seed is invalid or duplicated")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("ECRD score SHA-256 is malformed")
        matrix[arm][seed] = (Path(path_text), digest)
    if any(tuple(sorted(records)) != ECRD_MODEL_SEEDS for records in matrix.values()):
        raise ValueError("ECRD reducer requires the complete four-by-three matrix")
    return matrix


def _write_summary_csv(path: Path, gate: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    columns = (
        "arm",
        "overall_equal_field_fair_crps",
        "field_calibration_log_error",
        "spectral_power_log_error",
        "material_power_checks_passing",
        "ne_phi_cross_spectrum_error",
        "ne_phi_coherence_error",
        "ne_phi_phase_error_degrees",
        "spatial_transport_covariance_error",
        "integrated_transport_calibration_log_error",
        "median_integrated_transport_spread_skill",
        "eligible_candidate",
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for arm in ECRD_ARMS:
            summary = gate["summaries"][arm]
            cross = summary["Ne_phi_dependence"]["overall"]
            candidate = gate["eligible_candidate_gates"].get(arm)
            writer.writerow(
                {
                    "arm": arm,
                    "overall_equal_field_fair_crps": summary[
                        "seed_mean_overall_equal_field_fair_crps"
                    ],
                    "field_calibration_log_error": summary[
                        "median_absolute_log_field_spread_skill_error"
                    ],
                    "spectral_power_log_error": summary[
                        "material_spectral_power"
                    ]["median_absolute_log_power_ratio_error"],
                    "material_power_checks_passing": summary[
                        "material_spectral_power"
                    ]["passing_count"],
                    "ne_phi_cross_spectrum_error": cross[
                        "complex_cross_spectrum_relative_L1_error"
                    ],
                    "ne_phi_coherence_error": cross[
                        "truth_amplitude_weighted_absolute_coherence_error"
                    ],
                    "ne_phi_phase_error_degrees": cross[
                        "truth_amplitude_weighted_absolute_phase_error_degrees"
                    ],
                    "spatial_transport_covariance_error": summary[
                        "median_spatial_transport_covariance_relative_error"
                    ],
                    "integrated_transport_calibration_log_error": summary[
                        "median_absolute_log_integrated_transport_spread_skill_error"
                    ],
                    "median_integrated_transport_spread_skill": summary[
                        "median_integrated_transport_spread_skill_ratio"
                    ],
                    "eligible_candidate": (
                        ""
                        if candidate is None
                        else bool(candidate["all_seven_families_pass"])
                    ),
                }
            )


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    runtime_paths = (
        args.event_threshold_result,
        args.evaluation_manifest,
        args.output,
        *[item for item in args.score],
    )
    if any("85606" in str(path).lower() for path in runtime_paths):
        raise ValueError("held-out paths are prohibited during ECRD reduction")
    output = Path(args.output)
    assert_development_path(output)
    if output.exists():
        raise FileExistsError(output)
    matrix_paths = parse_score_specifications(args.score)
    manifest_path = verify_input(
        args.evaluation_manifest,
        args.evaluation_manifest_sha256,
        "ECRD evaluation manifest",
    )
    manifest = load_strict_json(manifest_path)
    if (
        manifest.get("status")
        != "frozen_after_ECRD_training_before_85604_scientific_evaluation"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("ECRD reducer manifest scope differs")
    threshold_path = verify_input(
        args.event_threshold_result,
        args.event_threshold_result_sha256,
        "ECRD event-threshold result",
    )
    if (
        manifest.get("evidence_locks", {})
        .get("event_threshold_result", {})
        .get("sha256")
        != args.event_threshold_result_sha256
    ):
        raise RuntimeError("ECRD reducer event-threshold lock differs")
    threshold = load_strict_json(threshold_path)
    materiality = threshold.get("spectral_materiality", {})
    validate_b2_spectral_materiality(materiality)

    scores: dict[str, dict[int, Mapping[str, Any]]] = {
        arm: {} for arm in ECRD_ARMS
    }
    score_inputs: dict[str, Any] = {}
    for arm in ECRD_ARMS:
        for seed in ECRD_MODEL_SEEDS:
            source, digest = matrix_paths[arm][seed]
            path = verify_input(source, digest, f"{arm} seed {seed} score")
            score = load_strict_json(path)
            if score.get("arm") != arm or score.get("model_seed") != seed:
                raise RuntimeError("ECRD score specification/contents differ")
            scores[arm][seed] = score
            score_inputs[f"{arm}:seed{seed}"] = {
                "path": str(path),
                "sha256": digest,
            }
    gate = evaluate_ecrd_model_ladder(scores, spectral_materiality=materiality)
    output.mkdir(parents=True)
    gate_path = output / "acceptance.json"
    write_strict_json_atomic(gate_path, gate)
    summary_path = output / "model_summary.csv"
    _write_summary_csv(summary_path, gate)
    result = {
        "schema_version": 1,
        "scope": "ECRD_model_ladder_final_85604_reduction",
        "status": (
            "candidate_selected_pending_explicit_holdout_release"
            if gate["selected_arm"] is not None
            else "no_candidate_passed_state_data_bottleneck"
        ),
        "development_run": "85604",
        "held_out_85606_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "score_inputs": score_inputs,
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "event_threshold_result": {
            "path": str(threshold_path),
            "sha256": args.event_threshold_result_sha256,
        },
        "acceptance": {
            "path": str(gate_path.resolve(strict=True)),
            "sha256": sha256_path(gate_path),
        },
        "model_summary": {
            "path": str(summary_path.resolve(strict=True)),
            "sha256": sha256_path(summary_path),
        },
        "selected_arm": gate["selected_arm"],
        "held_out_release_eligible": gate["held_out_release_eligible"],
        "held_out_85606_access_authorized": False,
        "physics_derived_training_loss_used": False,
        "assimilation_authorized": False,
        "diagnostic_ranking_authorized": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    index_path = output / "artifact_sha256.txt"
    index_path.write_text(
        "\n".join(
            f"{sha256_path(path)}  {path.resolve(strict=True)}"
            for path in (gate_path, summary_path, result_path)
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_arm": result["selected_arm"],
                "result": str(result_path.resolve(strict=True)),
                "result_sha256": sha256_path(result_path),
                "artifact_index_sha256": sha256_path(index_path),
                "held_out_85606_read": False,
                "held_out_85606_access_authorized": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
