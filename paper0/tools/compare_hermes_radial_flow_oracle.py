#!/usr/bin/env python3
"""Compare candidate total radial flow and divergence with compiled Hermes."""

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

from compare_hermes_xy_face_oracle import (  # noqa: E402
    CASES,
    comparison_regions as face_regions,
    field_input_metrics,
)
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
    divergence_from_radial_face_flow_partial,
    radial_exb_face_flow_partial,
    radial_exb_xz_face_flow_partial,
    toroidal_wedge_spacing,
)


FACE_QUANTITIES = ("xz_flow", "xy_flow", "total_radial_flow")
DEFAULT_ATOL = 5.0e-10
DEFAULT_RTOL = 5.0e-10
CONSERVATION_ATOL = 5.0e-12
CONSERVATION_RTOL = 5.0e-12
DZ_ATOL = 1.0e-15


def canonical_xy(variable: Any) -> np.ndarray:
    dimensions = list(variable.dimensions)
    values = np.ma.filled(variable[:], np.nan)
    values = np.asarray(values, dtype=np.float64)
    unknown = [name for name in dimensions if name not in {"x", "y"}]
    if unknown or sorted(dimensions) != ["x", "y"]:
        raise ValueError(f"{variable.name} must contain x and y exactly once")
    return np.transpose(values, [dimensions.index(axis) for axis in "xy"])


def assemble_xy_partitions(
    partitions: list[tuple[int, np.ndarray]],
    *,
    global_ny: int = 32,
) -> np.ndarray:
    if not partitions:
        raise ValueError("at least one xy partition is required")
    expanded = [(identifier, values[..., None]) for identifier, values in partitions]
    return assemble_y_partitions(expanded, global_ny=global_ny)[..., 0]


def cell_regions(cell_indices: np.ndarray) -> dict[str, np.ndarray]:
    cells = np.asarray(cell_indices, dtype=np.int64)[:, None]
    y = np.arange(32, dtype=np.int64)[None, :]
    valid = np.broadcast_to((y > 0) & (y < 31), (cells.size, 32)).copy()
    inside = cells < MODEL_SEPARATRIX_X
    private_flux = np.broadcast_to(
        inside & ((y == 7) | (y == 24)), valid.shape
    )
    core_branch = np.broadcast_to(
        inside & ((y == 8) | (y == 23)), valid.shape
    )
    separatrix = np.broadcast_to(
        (cells == MODEL_SEPARATRIX_X) & valid, valid.shape
    )
    open_sol = np.broadcast_to((cells >= MODEL_SEPARATRIX_X) & valid, valid.shape)
    ordinary = valid & ~private_flux & ~core_branch
    return {
        "all_valid": valid,
        "ordinary_or_sol_sequential": ordinary,
        "inner_private_flux_connection": private_flux,
        "inner_core_branch_connection": core_branch,
        "separatrix_cell": separatrix,
        "open_sol_interior": open_sol,
    }


def sign_counts(values: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    selected = values[np.broadcast_to(mask[..., None], values.shape)]
    finite = np.isfinite(selected)
    return {
        "nonfinite_count": int(selected.size - np.count_nonzero(finite)),
        "positive_count": int(np.count_nonzero(selected[finite] > 0.0)),
        "negative_count": int(np.count_nonzero(selected[finite] < 0.0)),
        "zero_count": int(np.count_nonzero(selected[finite] == 0.0)),
    }


def conservation_metrics(
    divergence: np.ndarray,
    face_flow: np.ndarray,
    jacobian: np.ndarray,
    dx: np.ndarray,
    cell_indices: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    denominator = jacobian[cell_indices] * dx[cell_indices]
    reconstructed_difference = divergence * denominator[..., None]
    direct_difference = face_flow[1:] - face_flow[:-1]
    return error_metrics(
        reconstructed_difference,
        direct_difference,
        mask,
        atol=CONSERVATION_ATOL,
        rtol=CONSERVATION_RTOL,
    )


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


def assembled_xyz(
    outputs: list[Any],
    decomposition: list[dict[str, Any]],
    name: str,
) -> np.ndarray:
    return assemble_y_partitions(
        [
            (metadata["PE_YIND"], canonical_xyz(output.variables[name]))
            for metadata, output in zip(decomposition, outputs)
        ]
    )


def main() -> int:
    args = parse_args()
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("tolerances must be nonnegative")
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("netCDF4 is required for the compiled oracle") from error

    with netCDF4.Dataset(args.grid, "r") as grid:
        geometry = {
            name: np.asarray(grid.variables[source][:], dtype=np.float64)[
                MODEL_X_SLICE
            ]
            for name, source in {
                "jacobian": "J",
                "dx": "dx",
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

        dz = assemble_xy_partitions(
            [
                (metadata["PE_YIND"], canonical_xy(output.variables["dz"]))
                for metadata, output in zip(decomposition, outputs)
            ]
        )[MODEL_X_SLICE]
        expected_dz = toroidal_wedge_spacing(81, zperiod=5)
        dz_max_abs_error = float(np.max(np.abs(dz - expected_dz)))
        dz_passed = bool(
            np.all(np.isfinite(dz)) and dz_max_abs_error <= DZ_ATOL
        )

        for case in CASES:
            q = assembled_xyz(outputs, decomposition, f"q_{case}")[MODEL_X_SLICE]
            phi = assembled_xyz(outputs, decomposition, f"phi_{case}")[
                MODEL_X_SLICE
            ]
            reference_full = {
                quantity: assembled_xyz(outputs, decomposition, f"{quantity}_{case}")[
                    MODEL_X_SLICE
                ]
                for quantity in FACE_QUANTITIES
            }
            reference_divergence_full = assembled_xyz(
                outputs, decomposition, f"radial_divergence_{case}"
            )[MODEL_X_SLICE]

            candidate_faces = radial_exb_face_flow_partial(
                q,
                phi,
                geometry["jacobian"],
                geometry["g11"],
                geometry["g23"],
                geometry["bxy"],
                geometry["z_shift"],
                geometry["dy"],
                shift_angle,
                dz=expected_dz,
                topology=TOPOLOGY,
                zperiod=5,
                positive=True,
            )
            candidate_divergence = (
                divergence_from_radial_face_flow_partial(
                    candidate_faces,
                    geometry["jacobian"],
                    dx=geometry["dx"],
                )
            )
            left_indices = candidate_faces.left_cell_indices
            cell_indices = candidate_divergence.cell_indices
            face_masks = face_regions(left_indices)
            cell_masks = cell_regions(cell_indices)

            reference_faces = {
                name: np.take(values, left_indices, axis=0)
                for name, values in reference_full.items()
            }
            reference_divergence = np.take(
                reference_divergence_full, cell_indices, axis=0
            )
            candidate_values = {
                "xz_flow": candidate_faces.xz_flow,
                "xy_flow": candidate_faces.xy_flow,
                "total_radial_flow": candidate_faces.flow,
            }
            face_metrics = {
                quantity: {
                    region_name: error_metrics(
                        candidate_values[quantity],
                        reference_faces[quantity],
                        region_mask,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    for region_name, region_mask in face_masks.items()
                }
                for quantity in FACE_QUANTITIES
            }
            divergence_metrics = {
                region_name: error_metrics(
                    candidate_divergence.divergence,
                    reference_divergence,
                    region_mask,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                for region_name, region_mask in cell_masks.items()
            }
            component_sum_reference = error_metrics(
                reference_faces["total_radial_flow"],
                reference_faces["xz_flow"] + reference_faces["xy_flow"],
                face_masks["all_valid"],
                atol=0.0,
                rtol=0.0,
            )
            component_sum_candidate = error_metrics(
                candidate_faces.flow,
                candidate_faces.xz_flow + candidate_faces.xy_flow,
                face_masks["all_valid"],
                atol=0.0,
                rtol=0.0,
            )
            conservation_reference = conservation_metrics(
                reference_divergence,
                reference_faces["total_radial_flow"],
                geometry["jacobian"],
                geometry["dx"],
                cell_indices,
                cell_masks["all_valid"],
            )
            conservation_candidate = conservation_metrics(
                candidate_divergence.divergence,
                candidate_faces.flow,
                geometry["jacobian"],
                geometry["dx"],
                cell_indices,
                cell_masks["all_valid"],
            )
            xz_signs = sign_counts(
                reference_faces["xz_flow"], face_masks["all_valid"]
            )
            total_signs = sign_counts(
                reference_faces["total_radial_flow"], face_masks["all_valid"]
            )
            coverage_passed = True
            if case != "constant":
                coverage_passed = all(
                    counts["nonfinite_count"] == 0
                    and counts["positive_count"] > 0
                    and counts["negative_count"] > 0
                    for counts in (xz_signs, total_signs)
                )
            input_metrics = field_input_metrics(q, phi, case)
            numerical_passed = all(
                metric["passed"]
                for quantity in face_metrics.values()
                for metric in quantity.values()
            ) and all(metric["passed"] for metric in divergence_metrics.values())
            structural_passed = all(
                item["passed"]
                for item in (
                    component_sum_reference,
                    component_sum_candidate,
                    conservation_reference,
                    conservation_candidate,
                )
            )
            passed = bool(
                input_metrics["passed"]
                and dz_passed
                and coverage_passed
                and numerical_passed
                and structural_passed
            )
            case_metrics[case] = {
                "passed": passed,
                "input_validation": input_metrics,
                "coverage_passed": bool(coverage_passed),
                "xz_flow_signs": xz_signs,
                "total_flow_signs": total_signs,
                "face_quantities": face_metrics,
                "radial_divergence": divergence_metrics,
                "exact_component_sum": {
                    "reference": component_sum_reference,
                    "candidate": component_sum_candidate,
                },
                "volume_weighted_conservation": {
                    "reference": conservation_reference,
                    "candidate": conservation_candidate,
                },
            }

            arrays[f"q_{case}"] = q
            arrays[f"phi_{case}"] = phi
            for quantity in FACE_QUANTITIES:
                arrays[f"oracle_{quantity}_{case}"] = reference_faces[quantity]
                arrays[f"candidate_{quantity}_{case}"] = candidate_values[quantity]
            arrays[f"oracle_radial_divergence_{case}"] = reference_divergence
            arrays[f"candidate_radial_divergence_{case}"] = (
                candidate_divergence.divergence
            )

    overall_passed = dz_passed and all(
        item["passed"] for item in case_metrics.values()
    )
    result = {
        "schema_version": 1,
        "phase": "phase2_hermes_combined_radial_flow_compiled_oracle",
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
            "safe_model_cell_indices": [2, 62],
            "evaluated_y_slice": [1, 31],
        },
        "native_dz_validation": {
            "expected": expected_dz,
            "maximum_absolute_error": dz_max_abs_error,
            "atol": DZ_ATOL,
            "passed": dz_passed,
        },
        "operator": {
            "reference": (
                "compiled GPL source-derived Hermes-3 xz plus shifted-xy "
                "radial face flow and divergence at revision 920ba829"
            ),
            "candidate": "tcv_diagnostics.transport.radial_exb_face_flow_partial",
            "hermes_source_ranges": [
                "src/div_ops.cxx:128-229",
                "src/div_ops.cxx:273-326",
            ],
            "mpi_decomposition": decomposition,
        },
        "acceptance_rule_frozen_before_execution": {
            "continuous_atol": args.atol,
            "continuous_rtol": args.rtol,
            "continuous_formula": (
                "max_abs_error <= continuous_atol + "
                "continuous_rtol * max_abs_reference"
            ),
            "conservation_atol": CONSERVATION_ATOL,
            "conservation_rtol": CONSERVATION_RTOL,
            "component_sum_requires_exact_equality": True,
            "requires_every_case_quantity_and_region": True,
            "requires_both_xz_and_total_flow_signs_in_nonconstant_cases": True,
        },
        "cases": case_metrics,
        "overall_passed": bool(overall_passed),
        "candidate_status": (
            "accepted_for_combined_radial_flow_stage"
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
