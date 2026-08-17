#!/usr/bin/env python3
"""Compare the shifted-xy face flow with a compiled Hermes oracle."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_shifted_ddy_oracle import (  # noqa: E402
    MODEL_SEPARATRIX_X,
    MODEL_X_SLICE,
    TOPOLOGY,
    assemble_y_partitions,
    canonical_xyz,
    error_metrics,
    scalar_integer,
    sha256_file,
)
from tcv_diagnostics.transport import (  # noqa: E402
    radial_exb_xy_face_flow_partial,
)


CASES = ("constant", "smooth", "signed", "clipping")
QUANTITIES = ("velocity", "state", "flow")
DEFAULT_ATOL = 5.0e-10
DEFAULT_RTOL = 5.0e-10


def comparison_regions(left_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Return frozen radial-face/y masks for the safe model domain."""

    left = np.asarray(left_indices, dtype=np.int64)[:, None]
    y = np.arange(32, dtype=np.int64)[None, :]
    valid = np.broadcast_to((y > 0) & (y < 31), (left.size, 32)).copy()
    branch_relevant = left < MODEL_SEPARATRIX_X
    private_flux = np.broadcast_to(
        branch_relevant & ((y == 7) | (y == 24)), valid.shape
    )
    core_branch = np.broadcast_to(
        branch_relevant & ((y == 8) | (y == 23)), valid.shape
    )
    separatrix_face = np.broadcast_to(
        (left == MODEL_SEPARATRIX_X - 1) & valid, valid.shape
    )
    open_sol = np.broadcast_to((left >= MODEL_SEPARATRIX_X) & valid, valid.shape)
    ordinary = valid & ~private_flux & ~core_branch
    return {
        "all_valid": valid,
        "ordinary_or_sol_sequential": ordinary,
        "inner_private_flux_connection": private_flux,
        "inner_core_branch_connection": core_branch,
        "separatrix_radial_face": separatrix_face,
        "open_sol_interior": open_sol,
    }


def field_input_metrics(
    q: np.ndarray,
    phi: np.ndarray,
    case: str,
) -> dict[str, Any]:
    """Reject missing, collapsed, or non-finite manufactured inputs."""

    metrics: dict[str, Any] = {}
    passed = True
    for name, values in (("q", q), ("phi", phi)):
        finite = np.isfinite(values)
        nonfinite_count = int(values.size - np.count_nonzero(finite))
        if np.any(finite):
            minimum = float(np.min(values[finite]))
            maximum = float(np.max(values[finite]))
            peak_to_peak = maximum - minimum
        else:
            minimum = float("inf")
            maximum = float("inf")
            peak_to_peak = 0.0
        expected = 2.5 if name == "q" else 4.0
        constant_error = None
        if case == "constant":
            constant_error = (
                float(np.max(np.abs(values[finite] - expected)))
                if np.any(finite)
                else float("inf")
            )
            item_passed = nonfinite_count == 0 and constant_error <= 1.0e-13
        else:
            item_passed = nonfinite_count == 0 and peak_to_peak > 1.0e-6
        metrics[name] = {
            "nonfinite_count": nonfinite_count,
            "minimum": minimum,
            "maximum": maximum,
            "peak_to_peak": peak_to_peak,
            "constant_max_abs_expected_error": constant_error,
            "passed": bool(item_passed),
        }
        passed = passed and item_passed
    metrics["passed"] = bool(passed)
    return metrics


def discrete_clip_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    expanded = np.broadcast_to(mask[..., None], reference.shape)
    reference_values = reference[expanded]
    candidate_values = candidate[expanded]
    finite = np.isfinite(reference_values)
    binary = finite & (
        (np.abs(reference_values) <= 1.0e-13)
        | (np.abs(reference_values - 1.0) <= 1.0e-13)
    )
    reference_bool = reference_values > 0.5
    mismatch = candidate_values.astype(bool) != reference_bool
    return {
        "point_count": int(reference_values.size),
        "reference_nonfinite_count": int(np.count_nonzero(~finite)),
        "reference_nonbinary_count": int(np.count_nonzero(~binary)),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "reference_clipped_count": int(np.count_nonzero(reference_bool)),
        "passed": bool(np.all(binary) and not np.any(mismatch)),
    }


def sign_coverage(velocity: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = velocity[np.broadcast_to(mask[..., None], velocity.shape)]
    finite = np.isfinite(values)
    return {
        "nonfinite_count": int(values.size - np.count_nonzero(finite)),
        "positive_count": int(np.count_nonzero(values[finite] > 0.0)),
        "negative_count": int(np.count_nonzero(values[finite] < 0.0)),
        "zero_count": int(np.count_nonzero(values[finite] == 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bout-output", type=Path, nargs="+", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    return parser.parse_args()


def assembled_field(
    outputs: list[Any],
    decomposition: list[dict[str, Any]],
    name: str,
) -> np.ndarray:
    return assemble_y_partitions(
        [
            (
                metadata["PE_YIND"],
                canonical_xyz(output.variables[name]),
            )
            for metadata, output in zip(decomposition, outputs)
        ]
    )


def main() -> int:
    args = parse_args()
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("tolerances must be nonnegative")

    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for the compiled oracle") from error

    with netCDF4.Dataset(args.grid, "r") as grid:
        geometry = {
            name: np.asarray(grid.variables[source][:], dtype=np.float64)[
                MODEL_X_SLICE
            ]
            for name, source in {
                "jacobian": "J",
                "g11": "g11",
                "g23": "g23",
                "bxy": "Bxy",
                "z_shift": "zShift",
                "dy": "dy",
            }.items()
        }
        shift_angle = np.asarray(
            grid.variables["ShiftAngle"][:], dtype=np.float64
        )[MODEL_X_SLICE]

    case_metrics: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    with ExitStack() as stack:
        outputs = [
            stack.enter_context(netCDF4.Dataset(path, "r"))
            for path in args.bout_output
        ]
        decomposition = []
        for path, output in zip(args.bout_output, outputs):
            metadata = {
                "path": str(path),
                "NXPE": scalar_integer(output, "NXPE"),
                "NYPE": scalar_integer(output, "NYPE"),
                "MYSUB": scalar_integer(output, "MYSUB"),
                "PE_XIND": scalar_integer(output, "PE_XIND"),
                "PE_YIND": scalar_integer(output, "PE_YIND"),
            }
            if metadata["NXPE"] != 1 or metadata["NYPE"] != 4:
                raise ValueError(f"expected NXPE=1,NYPE=4; got {metadata}")
            if metadata["MYSUB"] != 8 or metadata["PE_XIND"] != 0:
                raise ValueError(f"unexpected rank decomposition: {metadata}")
            decomposition.append(metadata)

        for case in CASES:
            q = assembled_field(outputs, decomposition, f"q_{case}")[MODEL_X_SLICE]
            phi = assembled_field(outputs, decomposition, f"phi_{case}")[
                MODEL_X_SLICE
            ]
            reference_full = {
                quantity: assembled_field(
                    outputs, decomposition, f"xy_{quantity}_{case}"
                )[MODEL_X_SLICE]
                for quantity in QUANTITIES
            }
            reference_clipped_full = assembled_field(
                outputs, decomposition, f"xy_clipped_{case}"
            )[MODEL_X_SLICE]

            candidate_result = radial_exb_xy_face_flow_partial(
                q,
                phi,
                geometry["jacobian"],
                geometry["g11"],
                geometry["g23"],
                geometry["bxy"],
                geometry["z_shift"],
                geometry["dy"],
                shift_angle,
                topology=TOPOLOGY,
                zperiod=5,
                positive=True,
            )
            left_indices = candidate_result.left_cell_indices
            regions = comparison_regions(left_indices)
            candidate = {
                "velocity": candidate_result.velocity_factor,
                "state": candidate_result.upwind_state,
                "flow": candidate_result.flow,
            }
            reference = {
                name: np.take(values, left_indices, axis=0)
                for name, values in reference_full.items()
            }
            reference_clipped = np.take(
                reference_clipped_full, left_indices, axis=0
            )

            quantity_metrics = {
                quantity: {
                    region_name: error_metrics(
                        candidate[quantity],
                        reference[quantity],
                        region_mask,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    for region_name, region_mask in regions.items()
                }
                for quantity in QUANTITIES
            }
            clip_metrics = discrete_clip_metrics(
                candidate_result.positivity_clipped_mask,
                reference_clipped,
                regions["all_valid"],
            )
            signs = sign_coverage(reference["velocity"], regions["all_valid"])
            input_metrics = field_input_metrics(q, phi, case)
            coverage_passed = True
            if case != "constant":
                coverage_passed = (
                    signs["nonfinite_count"] == 0
                    and signs["positive_count"] > 0
                    and signs["negative_count"] > 0
                )
            if case == "clipping":
                coverage_passed = (
                    coverage_passed
                    and clip_metrics["reference_clipped_count"] > 0
                    and clip_metrics["reference_clipped_count"]
                    < clip_metrics["point_count"]
                )
            numerical_passed = all(
                item["passed"]
                for by_region in quantity_metrics.values()
                for item in by_region.values()
            )
            passed = bool(
                input_metrics["passed"]
                and numerical_passed
                and clip_metrics["passed"]
                and coverage_passed
            )
            case_metrics[case] = {
                "passed": passed,
                "input_validation": input_metrics,
                "velocity_sign_coverage": signs,
                "coverage_passed": bool(coverage_passed),
                "quantities": quantity_metrics,
                "positivity_clip": clip_metrics,
            }

            arrays[f"q_{case}"] = q
            arrays[f"phi_{case}"] = phi
            for quantity in QUANTITIES:
                arrays[f"oracle_{quantity}_{case}"] = reference[quantity]
                arrays[f"candidate_{quantity}_{case}"] = candidate[quantity]
                arrays[f"difference_{quantity}_{case}"] = (
                    candidate[quantity] - reference[quantity]
                )
            arrays[f"oracle_clipped_{case}"] = reference_clipped
            arrays[f"candidate_clipped_{case}"] = (
                candidate_result.positivity_clipped_mask
            )

    overall_passed = all(item["passed"] for item in case_metrics.values())
    result = {
        "schema_version": 1,
        "phase": "phase2_hermes_shifted_xy_face_compiled_oracle",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": int(args.slurm_job_id),
        "data": {
            "run_id": "85604_geometry_only",
            "plasma_state_frames_read": 0,
            "held_out_85606_read": False,
            "native_z_samples": 81,
            "zperiod": 5,
            "model_x_grid_slice": [2, 66],
            "safe_model_left_face_indices": [1, 62],
            "evaluated_y_slice": [1, 31],
        },
        "operator": {
            "reference": (
                "compiled GPL source-derived Hermes-3 shifted-xy radial face "
                "operator at revision 920ba829"
            ),
            "candidate": (
                "tcv_diagnostics.transport."
                "radial_exb_xy_face_flow_partial"
            ),
            "hermes_source_file": "src/div_ops.cxx:273-326",
            "topology": {
                "model_separatrix_x_index": MODEL_SEPARATRIX_X,
                "core_connection": [8, 23],
                "private_flux_connection": [7, 24],
            },
            "mpi_decomposition": decomposition,
        },
        "acceptance_rule_frozen_before_execution": {
            "atol": args.atol,
            "rtol": args.rtol,
            "formula": "max_abs_error <= atol + rtol * max_abs_reference",
            "requires_no_nonfinite_values": True,
            "requires_every_case_quantity_and_region": True,
            "requires_exact_binary_clipping_decisions": True,
            "requires_both_velocity_signs_in_nonconstant_cases": True,
            "requires_selected_clipped_and_unclipped_states_in_clipping_case": True,
        },
        "cases": case_metrics,
        "overall_passed": overall_passed,
        "candidate_status": (
            "accepted_for_shifted_xy_face_stage"
            if overall_passed
            else "rejected_pending_debug"
        ),
        "artifacts": {
            "bout_outputs": [str(path) for path in args.bout_output],
            "bout_output_digests": {
                str(path): sha256_file(path) for path in args.bout_output
            },
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
