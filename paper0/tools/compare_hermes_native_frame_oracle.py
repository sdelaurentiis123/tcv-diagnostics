#!/usr/bin/env python3
"""Compare selected native-81 85604 states with compiled Hermes flow."""

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

from compare_hermes_radial_flow_oracle import (  # noqa: E402
    CONSERVATION_ATOL,
    CONSERVATION_RTOL,
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    DZ_ATOL,
    FACE_QUANTITIES,
    assemble_xy_partitions,
    assembled_xyz,
    canonical_xy,
    cell_regions,
    conservation_metrics,
)
from compare_hermes_xy_face_oracle import (  # noqa: E402
    comparison_regions as face_regions,
)
from compare_shifted_ddy_oracle import (  # noqa: E402
    MODEL_X_SLICE,
    TOPOLOGY,
    error_metrics,
    scalar_integer,
    sha256_file,
)
from tcv_diagnostics.transport import (  # noqa: E402
    divergence_from_radial_face_flow_partial,
    radial_exb_face_flow_partial,
    toroidal_wedge_spacing,
)


FRAME_INDICES = (0, 156, 312, 467, 623)
ADVECTED_FIELDS = ("Ne", "Pe", "Pi")
CLOSURE_ATOL = 1.0e-12
CLOSURE_RTOL = 1.0e-12
INPUT_RANGE_MINIMUM = 1.0e-12
FLOW_MAXIMUM_MINIMUM = 1.0e-12


def frame_label(frame: int) -> str:
    return f"f{frame:03d}"


def dynamic_input_metrics(values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    nonfinite_count = int(values.size - np.count_nonzero(finite))
    if np.any(finite):
        minimum = float(np.min(values[finite]))
        maximum = float(np.max(values[finite]))
        peak_to_peak = float(maximum - minimum)
    else:
        minimum = float("inf")
        maximum = float("inf")
        peak_to_peak = 0.0
    return {
        "nonfinite_count": nonfinite_count,
        "minimum": minimum,
        "maximum": maximum,
        "peak_to_peak": peak_to_peak,
        "minimum_required_peak_to_peak": INPUT_RANGE_MINIMUM,
        "passed": bool(
            nonfinite_count == 0 and peak_to_peak > INPUT_RANGE_MINIMUM
        ),
    }


def maximum_absolute_metrics(
    values: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    selected = values[np.broadcast_to(mask[..., None], values.shape)]
    finite = np.isfinite(selected)
    nonfinite_count = int(selected.size - np.count_nonzero(finite))
    maximum = float(np.max(np.abs(selected[finite]))) if np.any(finite) else 0.0
    return {
        "point_count": int(selected.size),
        "nonfinite_count": nonfinite_count,
        "maximum_absolute": maximum,
        "minimum_required_maximum_absolute": FLOW_MAXIMUM_MINIMUM,
        "passed": bool(
            nonfinite_count == 0 and maximum > FLOW_MAXIMUM_MINIMUM
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--extraction-record", type=Path, required=True)
    parser.add_argument("--bout-output", type=Path, nargs="+", required=True)
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
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("netCDF4 is required for the native-frame oracle") from error

    extraction = json.loads(args.extraction_record.read_text(encoding="utf-8"))
    if extraction["development_run"] != "85604" or extraction["held_out_run_read"]:
        raise ValueError("extraction record violates the development-only scope")
    if extraction["frame_indices"] != list(FRAME_INDICES):
        raise ValueError("extraction record does not contain the frozen frames")
    if extraction["canonical_file_sha256"] != sha256_file(args.canonical):
        raise ValueError("canonical frame file hash differs from extraction record")

    with netCDF4.Dataset(args.canonical, "r") as canonical_file:
        dimensions = {
            name: int(len(dimension))
            for name, dimension in canonical_file.dimensions.items()
        }
        if dimensions != {"selected_frame": 5, "x": 64, "y": 32, "z": 81}:
            raise ValueError(f"unexpected canonical dimensions: {dimensions}")
        canonical_frames = np.asarray(
            canonical_file.variables["frame_index"][:], dtype=np.int64
        )
        if not np.array_equal(canonical_frames, FRAME_INDICES):
            raise ValueError("canonical file frame indices differ from protocol")
        canonical = {
            name: np.asarray(canonical_file.variables[name][:], dtype=np.float64)
            for name in ("Ne", "Ni", "Te", "Ti", "Pe", "Pi", "phi")
        }

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

    all_cells = np.ones((64, 32), dtype=bool)
    closure_relations = {
        "Ni_equals_Ne": (canonical["Ni"], canonical["Ne"]),
        "Pe_equals_Ne_times_Te": (
            canonical["Ne"] * canonical["Te"],
            canonical["Pe"],
        ),
        "Pi_equals_Ni_times_Ti": (
            canonical["Ni"] * canonical["Ti"],
            canonical["Pi"],
        ),
        "Pi_equals_Ne_times_Ti": (
            canonical["Ne"] * canonical["Ti"],
            canonical["Pi"],
        ),
    }
    closure_metrics = {
        relation: {
            frame_label(frame): error_metrics(
                candidate[position],
                reference[position],
                all_cells,
                atol=CLOSURE_ATOL,
                rtol=CLOSURE_RTOL,
            )
            for position, frame in enumerate(FRAME_INDICES)
        }
        for relation, (candidate, reference) in closure_relations.items()
    }
    closure_passed = all(
        metrics["passed"]
        for relation in closure_metrics.values()
        for metrics in relation.values()
    )

    case_metrics: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {
        f"canonical_{name}": values for name, values in canonical.items()
    }
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

        for position, frame in enumerate(FRAME_INDICES):
            label = frame_label(frame)
            for output in outputs:
                if scalar_integer(output, f"canonical_frame_index_{label}") != frame:
                    raise ValueError(f"compiled output frame marker differs for {label}")
            compiled_phi = assembled_xyz(
                outputs, decomposition, f"phi_{label}"
            )[MODEL_X_SLICE]
            phi = canonical["phi"][position]
            phi_echo = error_metrics(
                compiled_phi, phi, all_cells, atol=0.0, rtol=0.0
            )

            for field in ADVECTED_FIELDS:
                case = f"{field}_{label}"
                q = canonical[field][position]
                compiled_q = assembled_xyz(
                    outputs, decomposition, f"q_{case}"
                )[MODEL_X_SLICE]
                q_echo = error_metrics(
                    compiled_q, q, all_cells, atol=0.0, rtol=0.0
                )
                reference_full = {
                    quantity: assembled_xyz(
                        outputs, decomposition, f"{quantity}_{case}"
                    )[MODEL_X_SLICE]
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
                candidate_divergence = divergence_from_radial_face_flow_partial(
                    candidate_faces,
                    geometry["jacobian"],
                    dx=geometry["dx"],
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
                        region: error_metrics(
                            candidate_values[quantity],
                            reference_faces[quantity],
                            mask,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        for region, mask in face_masks.items()
                    }
                    for quantity in FACE_QUANTITIES
                }
                divergence_metrics = {
                    region: error_metrics(
                        candidate_divergence.divergence,
                        reference_divergence,
                        mask,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    for region, mask in cell_masks.items()
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
                q_input = dynamic_input_metrics(q)
                phi_input = dynamic_input_metrics(phi)
                flow_noncollapse = maximum_absolute_metrics(
                    reference_faces["total_radial_flow"],
                    face_masks["all_valid"],
                )
                numerical_passed = all(
                    metric["passed"]
                    for quantity in face_metrics.values()
                    for metric in quantity.values()
                ) and all(
                    metric["passed"] for metric in divergence_metrics.values()
                )
                structural_passed = all(
                    metric["passed"]
                    for metric in (
                        component_sum_reference,
                        component_sum_candidate,
                        conservation_reference,
                        conservation_candidate,
                    )
                )
                passed = bool(
                    dz_passed
                    and q_echo["passed"]
                    and phi_echo["passed"]
                    and q_input["passed"]
                    and phi_input["passed"]
                    and flow_noncollapse["passed"]
                    and numerical_passed
                    and structural_passed
                )
                case_metrics[case] = {
                    "passed": passed,
                    "frame_index": frame,
                    "advected_field": field,
                    "canonical_input_echo": {"q": q_echo, "phi": phi_echo},
                    "input_validation": {"q": q_input, "phi": phi_input},
                    "total_flow_noncollapse": flow_noncollapse,
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
                arrays[f"oracle_total_radial_flow_{case}"] = reference_faces[
                    "total_radial_flow"
                ]
                arrays[f"candidate_total_radial_flow_{case}"] = candidate_faces.flow
                arrays[f"oracle_radial_divergence_{case}"] = reference_divergence
                arrays[f"candidate_radial_divergence_{case}"] = (
                    candidate_divergence.divergence
                )

    overall_passed = bool(
        closure_passed
        and dz_passed
        and all(case["passed"] for case in case_metrics.values())
    )
    result = {
        "schema_version": 1,
        "phase": "phase2_hermes_native_85604_frame_oracle",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": int(args.slurm_job_id),
        "data": {
            "run_id": "85604",
            "frame_indices": list(FRAME_INDICES),
            "plasma_state_frames_read": len(FRAME_INDICES),
            "held_out_run_read": False,
            "native_z_samples": 81,
            "zperiod": 5,
            "model_x_grid_slice": [2, 66],
            "safe_model_left_face_indices": [1, 62],
            "safe_model_cell_indices": [2, 62],
            "evaluated_y_slice": [1, 31],
        },
        "extraction": {
            "record": str(args.extraction_record),
            "record_sha256": sha256_file(args.extraction_record),
            "canonical": str(args.canonical),
            "canonical_sha256": sha256_file(args.canonical),
        },
        "native_dz_validation": {
            "expected": expected_dz,
            "maximum_absolute_error": dz_max_abs_error,
            "atol": DZ_ATOL,
            "passed": dz_passed,
        },
        "five_channel_closure": {
            "atol": CLOSURE_ATOL,
            "rtol": CLOSURE_RTOL,
            "relations": closure_metrics,
            "passed": bool(closure_passed),
        },
        "operator": {
            "reference": (
                "compiled GPL source-derived Hermes-3 native-frame radial "
                "flow and divergence at revision 920ba829"
            ),
            "candidate": "tcv_diagnostics.transport.radial_exb_face_flow_partial",
            "advected_fields": list(ADVECTED_FIELDS),
            "hermes_source_ranges": [
                "src/div_ops.cxx:128-229",
                "src/div_ops.cxx:273-326",
            ],
            "mpi_decomposition": decomposition,
        },
        "acceptance_rule_frozen_before_execution": {
            "continuous_atol": args.atol,
            "continuous_rtol": args.rtol,
            "conservation_atol": CONSERVATION_ATOL,
            "conservation_rtol": CONSERVATION_RTOL,
            "canonical_input_echo_requires_exact_equality": True,
            "component_sum_requires_exact_equality": True,
            "requires_every_frame_advected_field_quantity_and_region": True,
            "input_peak_to_peak_minimum": INPUT_RANGE_MINIMUM,
            "total_flow_maximum_absolute_minimum": FLOW_MAXIMUM_MINIMUM,
        },
        "cases": case_metrics,
        "derived_internal_energy_scope": {
            "electron": "1.5 * face_flow(Pe, phi)",
            "ion": "1.5 * face_flow(Pi, phi)",
            "released_as_total_heat_flux": False,
        },
        "overall_passed": overall_passed,
        "candidate_status": (
            "accepted_for_selected_native_85604_frames"
            if overall_passed
            else "rejected_pending_documented_debug"
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
