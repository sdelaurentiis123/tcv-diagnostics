#!/usr/bin/env python3
"""Compare the Paper 0 shifted-DDY candidate with compiled BOUT++ output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.transport import (  # noqa: E402
    SingleNullTopology,
    shifted_ddy_single_null_partial,
)


CASES = ("constant", "zmode", "ycode", "mixed")
MODEL_X_SLICE = slice(2, 66)
MODEL_SEPARATRIX_X = 16
TOPOLOGY = SingleNullTopology(
    separatrix_x_index=MODEL_SEPARATRIX_X,
    core_lower_y=8,
    core_upper_y=23,
    pfr_lower_y=7,
    pfr_upper_y=24,
)
DEFAULT_ATOL = 5.0e-10
DEFAULT_RTOL = 5.0e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_xyz(variable: Any, *, expected_z: int = 81) -> np.ndarray:
    """Read a netCDF variable into canonical ``[x, y, z]`` order."""

    dimensions = list(variable.dimensions)
    values = np.ma.filled(variable[:], np.nan)
    values = np.asarray(values, dtype=np.float64)
    if "t" in dimensions:
        time_axis = dimensions.index("t")
        values = np.take(values, -1, axis=time_axis)
        dimensions.pop(time_axis)
    unknown = [name for name in dimensions if name not in {"x", "y", "z"}]
    if unknown:
        raise ValueError(f"unexpected dimensions {unknown} for {variable.name}")
    if sorted(dimensions) != ["x", "y", "z"]:
        raise ValueError(
            f"{variable.name} must contain x, y, and z exactly once; got {dimensions}"
        )
    values = np.transpose(values, [dimensions.index(axis) for axis in "xyz"])
    if values.shape[-1] != expected_z:
        raise ValueError(
            f"{variable.name} has {values.shape[-1]} z cells, expected {expected_z}"
        )
    return values


def strip_bout_y_guards(values: np.ndarray, *, physical_ny: int = 32) -> np.ndarray:
    """Remove symmetric BOUT++ y guards while refusing ambiguous shapes."""

    if values.ndim != 3:
        raise ValueError("BOUT field must have canonical [x, y, z] axes")
    extra = values.shape[1] - physical_ny
    if extra < 0 or extra % 2:
        raise ValueError(
            f"cannot symmetrically map output y={values.shape[1]} to ny={physical_ny}"
        )
    guard = extra // 2
    if guard == 0:
        return values
    return values[:, guard:-guard, :]


def comparison_regions() -> dict[str, np.ndarray]:
    x = np.arange(64, dtype=np.int64)[:, None]
    y = np.arange(32, dtype=np.int64)[None, :]
    valid = np.broadcast_to((y > 0) & (y < 31), (64, 32)).copy()
    inside = x < MODEL_SEPARATRIX_X
    pfr = np.broadcast_to(inside & ((y == 7) | (y == 24)), valid.shape)
    core_branch = np.broadcast_to(
        inside & ((y == 8) | (y == 23)), valid.shape
    )
    open_sol = np.broadcast_to((x >= MODEL_SEPARATRIX_X) & valid, valid.shape)
    ordinary = valid & ~pfr & ~core_branch
    return {
        "all_valid": valid,
        "ordinary_or_sol_sequential": ordinary,
        "inner_private_flux_connection": pfr,
        "inner_core_branch_connection": core_branch,
        "open_sol_interior": open_sol,
    }


def error_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    cell_mask: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch {candidate.shape} versus {reference.shape}")
    mask = np.broadcast_to(cell_mask[..., None], reference.shape)
    count = int(np.count_nonzero(mask))
    if count == 0:
        raise ValueError("comparison region is empty")
    candidate_values = candidate[mask]
    reference_values = reference[mask]
    finite = np.isfinite(candidate_values) & np.isfinite(reference_values)
    nonfinite_count = int(count - np.count_nonzero(finite))
    if np.any(finite):
        difference = candidate_values[finite] - reference_values[finite]
        max_abs_error = float(np.max(np.abs(difference)))
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        reference_scale = float(np.max(np.abs(reference_values[finite])))
    else:
        max_abs_error = float("inf")
        rmse = float("inf")
        reference_scale = float("inf")
    tolerance = float(atol + rtol * reference_scale)
    passed = bool(nonfinite_count == 0 and max_abs_error <= tolerance)
    return {
        "point_count": count,
        "nonfinite_count": nonfinite_count,
        "max_abs_reference": reference_scale,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "acceptance_tolerance": tolerance,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bout-output", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("tolerances must be nonnegative")

    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for the compiled oracle") from error

    with netCDF4.Dataset(args.grid, "r") as grid:
        z_shift = np.asarray(grid.variables["zShift"][:], dtype=np.float64)[
            MODEL_X_SLICE
        ]
        dy = np.asarray(grid.variables["dy"][:], dtype=np.float64)[MODEL_X_SLICE]
        shift_angle = np.asarray(
            grid.variables["ShiftAngle"][:], dtype=np.float64
        )[MODEL_X_SLICE]
        geometry_shapes = {
            "zShift": list(z_shift.shape),
            "dy": list(dy.shape),
            "ShiftAngle": list(shift_angle.shape),
        }

    regions = comparison_regions()
    case_metrics: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(args.bout_output, "r") as output:
        for case in CASES:
            field = strip_bout_y_guards(canonical_xyz(output.variables[f"input_{case}"]))
            reference = strip_bout_y_guards(
                canonical_xyz(output.variables[f"ddy_{case}"])
            )
            if field.shape[:2] != (68, 32) or reference.shape[:2] != (68, 32):
                raise ValueError(
                    f"unexpected BOUT interior shape for {case}: "
                    f"{field.shape}, {reference.shape}"
                )
            field = field[MODEL_X_SLICE]
            reference = reference[MODEL_X_SLICE]
            candidate_result = shifted_ddy_single_null_partial(
                field,
                z_shift,
                dy,
                shift_angle,
                topology=TOPOLOGY,
                zperiod=5,
            )
            candidate = candidate_result.values
            metrics_by_region = {
                name: error_metrics(
                    candidate,
                    reference,
                    mask,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                for name, mask in regions.items()
            }
            case_passed = all(item["passed"] for item in metrics_by_region.values())
            case_metrics[case] = {
                "passed": case_passed,
                "regions": metrics_by_region,
            }
            arrays[f"input_{case}"] = field
            arrays[f"bout_ddy_{case}"] = reference
            arrays[f"candidate_ddy_{case}"] = candidate
            arrays[f"difference_{case}"] = candidate - reference

    overall_passed = all(case["passed"] for case in case_metrics.values())
    result = {
        "schema_version": 1,
        "phase": "phase2_shifted_ddy_compiled_oracle",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": int(args.slurm_job_id),
        "data": {
            "run_id": "85604_geometry_only",
            "plasma_state_frames_read": 0,
            "held_out_85606_read": False,
            "native_z_samples": 81,
            "zperiod": 5,
            "model_x_grid_slice": [2, 66],
            "physical_y_slice": [0, 32],
            "evaluated_y_slice": [1, 31],
        },
        "operator": {
            "reference": "compiled BOUT++ 5.2.1 DDY with C2 and shifted transform",
            "candidate": "tcv_diagnostics.transport.shifted_ddy_single_null_partial",
            "topology": {
                "model_separatrix_x_index": MODEL_SEPARATRIX_X,
                "core_connection": [8, 23],
                "private_flux_connection": [7, 24],
            },
            "geometry_shapes": geometry_shapes,
        },
        "acceptance_rule_frozen_before_execution": {
            "atol": args.atol,
            "rtol": args.rtol,
            "formula": "max_abs_error <= atol + rtol * max_abs_reference",
            "requires_no_nonfinite_values": True,
            "requires_every_case_and_region": True,
        },
        "cases": case_metrics,
        "overall_passed": overall_passed,
        "candidate_status": "accepted_for_shifted_ddy_stage"
        if overall_passed
        else "rejected_pending_debug",
        "artifacts": {
            "bout_output": str(args.bout_output),
            "bout_output_digest": sha256_file(args.bout_output),
            "geometry_grid": str(args.grid),
            "geometry_grid_digest": sha256_file(args.grid),
            "comparison_arrays": str(args.arrays),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

