#!/usr/bin/env python3
"""Validate and summarize the locked Phase 0 legacy DA reproduction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_FREE = 0.2011699302399412
EXPECTED_ANCHORED = 0.17706738111186535


def _allowed(path: Path) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    text = str(resolved).lower()
    return "85606" not in text and all(part.lower() != "test" for part in resolved.parts)


def _require_input(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve(strict=True)
    if not _allowed(path):
        raise ValueError(f"refusing sequestered path: {path}")
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    return path


def _check_equal(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    passed = actual == expected
    return {"name": name, "actual": actual, "expected": expected, "passed": passed}


def _check_close(
    name: str, actual: float, expected: float, atol: float = 1e-12
) -> dict[str, Any]:
    passed = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)
    return {
        "name": name,
        "actual": float(actual),
        "expected": float(expected),
        "atol": atol,
        "passed": passed,
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric series")
    return float(sum(float(value) for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--numerical-atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.numerical_atol <= 0:
        raise ValueError("numerical-atol must be positive")
    summary_path = _require_input(args.summary)
    command_path = _require_input(args.command_file)
    output = Path(args.output).expanduser().resolve(strict=False)
    if not _allowed(output):
        raise ValueError(f"refusing sequestered output path: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol_checks = [
        _check_equal("dataset_split", summary.get("dataset_split"), "valid"),
        _check_equal("trajectory_index", summary.get("trajectory_index"), 0),
        _check_equal("trajectory_start", summary.get("trajectory_start"), 24),
        _check_equal("horizon", summary.get("horizon"), 48),
        _check_equal("context", summary.get("context"), 1),
        _check_equal("update", summary.get("update"), "etkf"),
        _check_equal("analysis_mode", summary.get("analysis_mode"), "filter"),
        _check_equal("layout", summary.get("layout"), "iter"),
        _check_equal("n_probes", summary.get("n_probes"), 69),
        _check_equal("samples", summary.get("samples"), 64),
        _check_equal("ensemble_M", summary.get("ensemble_M"), 64),
        _check_equal("assim_every", summary.get("assim_every"), 4),
        _check_equal("inflate", summary.get("inflate"), "off"),
        _check_equal(
            "observation_noise_policy",
            summary.get("observation_noise_policy"),
            "frame_keyed_common_random_numbers",
        ),
        _check_close("inflation", summary.get("inflation"), 1.0),
        _check_close("obs_std", summary.get("obs_std"), 0.05),
        _check_equal("fields", summary.get("fields"), ["Ne", "Te", "Ti", "phi", "Vi"]),
        _check_equal("is_oracle_analysis", summary.get("is_oracle_analysis"), False),
        _check_equal("crn_paired", summary.get("crn_paired"), True),
    ]
    failed = [check for check in protocol_checks if not check["passed"]]
    if failed:
        raise RuntimeError(f"locked reproduction protocol mismatch: {failed}")

    context = int(summary["context"])
    free_recomputed = _mean(summary["err_free"][context:])
    anchored_recomputed = _mean(summary["err_anchored"][context:])
    internal_checks = [
        _check_close("mean_err_free", summary["mean_err_free"], free_recomputed),
        _check_close(
            "mean_err_anchored", summary["mean_err_anchored"], anchored_recomputed
        ),
    ]
    failed_internal = [check for check in internal_checks if not check["passed"]]
    if failed_internal:
        raise RuntimeError(f"summary metric inconsistency: {failed_internal}")

    free_delta = free_recomputed - EXPECTED_FREE
    anchored_delta = anchored_recomputed - EXPECTED_ANCHORED
    numerical_checks = [
        _check_close(
            "historical_free_mean_rmse",
            free_recomputed,
            EXPECTED_FREE,
            atol=args.numerical_atol,
        ),
        _check_close(
            "historical_anchored_mean_rmse",
            anchored_recomputed,
            EXPECTED_ANCHORED,
            atol=args.numerical_atol,
        ),
    ]
    within_tolerance = all(check["passed"] for check in numerical_checks)
    benefit = free_recomputed - anchored_recomputed

    result = {
        "schema_version": "0.1.0",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "source_summary": str(summary_path),
        "exact_command": command_path.read_text(encoding="utf-8").strip(),
        "data_scope": "85604 legacy validation only",
        "blind_test_accessed": False,
        "diagnostic_class": "idealized direct-state oracle/control",
        "reporting_space": "legacy standardized five-field model space",
        "protocol_checks": protocol_checks,
        "internal_metric_checks": internal_checks,
        "historical_reference": {
            "free_mean_rmse": EXPECTED_FREE,
            "anchored_mean_rmse": EXPECTED_ANCHORED,
            "numerical_atol": args.numerical_atol,
        },
        "reproduction": {
            "free_mean_rmse": free_recomputed,
            "anchored_mean_rmse": anchored_recomputed,
            "absolute_rmse_improvement": benefit,
            "relative_rmse_improvement_percent": 100.0 * benefit / free_recomputed,
            "free_delta_from_history": free_delta,
            "anchored_delta_from_history": anchored_delta,
            "numerical_checks": numerical_checks,
            "status": "within_tolerance" if within_tolerance else "numerical_discrepancy",
        },
        "interpretation_limit": (
            "This reproduces a legacy idealized direct-state ETKF control. It does not "
            "validate a physical ITER diagnostic, transport fidelity, or held-out 85606 behavior."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Phase 0 reproduction: {result['reproduction']['status']} | "
        f"free={free_recomputed:.6f} anchored={anchored_recomputed:.6f} "
        f"improvement={benefit:.6f}"
    )


if __name__ == "__main__":
    main()
