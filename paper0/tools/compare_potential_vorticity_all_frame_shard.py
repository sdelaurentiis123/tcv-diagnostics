#!/usr/bin/env python3
"""Validate one frozen 78-frame potential/vorticity closure shard."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for search_path in (SRC, TOOLS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import compare_potential_elliptic_oracle as base  # noqa: E402
import compare_potential_elliptic_runtime_pressure_oracle as corrected  # noqa: E402
import compare_potential_vorticity_forward_oracle as selected  # noqa: E402


FRAME_COUNT = 78
SHARD_INTERVALS = tuple((start, start + FRAME_COUNT) for start in range(0, 624, 78))
INPUT_FIELDS = ("Ne", "Pe", "Pi", "Vort", "phi")
EXPECTED_SHAPE = (FRAME_COUNT, 64, 32, 81)
RHO_S0_METERS = 0.0007224847664314034
PRESSURE_DENOMINATOR = 3672.0
PRESSURE_FLOOR = 1.0e-7
CONSTANT_NULL_VALUE = 3.25
GAUGE_SHIFT = 7.0
MANUFACTURED_MODE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--extraction-record", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--bout-output", type=Path, nargs="+", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value} in {path}")
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_strict_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def verify_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    if (
        manifest["protocol_status"]
        != "frozen_before_first_all_624_frame_forward_closure_calculation"
        or manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
        or manifest["training_allowed"]
    ):
        raise ValueError("all-frame manifest violates the frozen scope")
    intervals = [tuple(item) for item in manifest["shards"]["half_open_intervals"]]
    if tuple(intervals) != SHARD_INTERVALS:
        raise ValueError("all-frame shard intervals differ")
    if not 0 <= args.shard_index < len(SHARD_INTERVALS):
        raise ValueError("shard index is outside the frozen range")
    protocol_lock = manifest["protocol"]
    expected_protocol = ROOT / protocol_lock["path"]
    if args.protocol.resolve() != expected_protocol.resolve():
        raise ValueError("all-frame protocol path differs")
    if sha256_file(args.protocol) != protocol_lock["sha256"]:
        raise ValueError("all-frame protocol SHA-256 differs")
    for name, lock in manifest["provenance_locks"].items():
        path = ROOT / lock["path"]
        if sha256_file(path) != lock["sha256"]:
            raise ValueError(f"predecessor SHA-256 differs for {name}")
    expected_grid = Path(manifest["raw_archive"]["root"]) / "tcv_85604_adjusted.nc"
    if args.grid.resolve() != expected_grid.resolve():
        raise ValueError("geometry path differs from the frozen archive")
    if sha256_file(args.grid) != manifest["raw_archive"]["geometry_sha256"]:
        raise ValueError("geometry SHA-256 differs")
    gate = manifest["source_forward_gate"]
    if args.atol != float(gate["atol"]) or args.rtol != float(gate["rtol"]):
        raise ValueError("source forward tolerance differs")
    return manifest


def validate_extraction(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    record = load_json(args.extraction_record)
    if (
        record["phase"]
        != "phase2_potential_vorticity_all_frame_85604_extraction"
        or record["development_run"] != "85604"
        or record["held_out_85606_read"]
        or record["training_performed"]
        or record["paper0_commit"] != args.paper0_commit
        or record["slurm_job_id"] != args.slurm_job_id
    ):
        raise ValueError("all-frame extraction provenance differs")
    if record["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("extraction manifest SHA-256 differs")
    if record["frame_indices"] != list(range(624)):
        raise ValueError("extraction does not cover all 624 frames")
    if record["shard_intervals"] != [list(item) for item in SHARD_INTERVALS]:
        raise ValueError("extraction shard intervals differ")
    if len(record["shards"]) != 8:
        raise ValueError("extraction must contain exactly eight shards")
    shard = record["shards"][args.shard_index]
    start, stop = SHARD_INTERVALS[args.shard_index]
    if (
        shard["shard_index"] != args.shard_index
        or shard["start"] != start
        or shard["stop"] != stop
        or shard["frame_indices"] != list(range(start, stop))
    ):
        raise ValueError("extraction shard identity differs")
    if args.canonical.resolve() != Path(shard["canonical_file"]).resolve():
        raise ValueError("canonical shard path differs from extraction")
    if sha256_file(args.canonical) != shard["canonical_file_sha256"]:
        raise ValueError("canonical shard SHA-256 differs")
    if record["raw_pressure_identity"] != {
        **manifest["raw_pressure_identity"],
        "negative_raw_Pe_count_by_shard": [0] * 8,
    }:
        raise ValueError("extracted raw-pressure inventory differs")
    for side in ("inner", "outer"):
        checks = record["boundary_checks"][side]
        if any(
            checks[name] != 0
            for name in (
                "outer_guard_copy_discrepancy_count",
                "midpoint_constancy_discrepancy_count",
                "nonfinite_count",
            )
        ):
            raise ValueError("extracted boundary structural gate fails")
    return record, shard, start, stop


def selected_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    cell_mask = np.asarray(mask)
    if cell_mask.dtype != np.bool_ or cell_mask.shape != array.shape[-3:-1]:
        raise ValueError("sufficient-statistic mask differs from spatial axes")
    shaped = cell_mask.reshape((1,) * (array.ndim - 3) + cell_mask.shape + (1,))
    return array[np.broadcast_to(shaped, array.shape)]


def sufficient_statistics(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    *,
    frame_start: int,
    location_axes: tuple[str, ...],
) -> dict[str, Any]:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape:
        raise ValueError("sufficient-statistic array shapes differ")
    candidate_values = selected_values(candidate_array, mask)
    reference_values = selected_values(reference_array, mask)
    finite = np.isfinite(candidate_values) & np.isfinite(reference_values)
    candidate_finite = candidate_values[finite]
    reference_finite = reference_values[finite]
    difference = candidate_finite - reference_finite
    metrics = base.continuous_metrics(
        candidate_array,
        reference_array,
        mask,
        location_axes=location_axes,
    )
    location = metrics["maximum_location"]
    if location is not None and "shard_frame_position" in location:
        location = dict(location)
        location["frame_index"] = frame_start + location.pop(
            "shard_frame_position"
        )
    return {
        "point_count": int(candidate_values.size),
        "finite_count": int(np.count_nonzero(finite)),
        "nonfinite_count": int(candidate_values.size - np.count_nonzero(finite)),
        "sum_candidate": float(np.sum(candidate_finite, dtype=np.float64)),
        "sum_reference": float(np.sum(reference_finite, dtype=np.float64)),
        "sum_candidate_squared": float(
            np.sum(candidate_finite * candidate_finite, dtype=np.float64)
        ),
        "sum_reference_squared": float(
            np.sum(reference_finite * reference_finite, dtype=np.float64)
        ),
        "sum_cross": float(
            np.sum(candidate_finite * reference_finite, dtype=np.float64)
        ),
        "sum_error": float(np.sum(difference, dtype=np.float64)),
        "sum_error_squared": float(
            np.sum(difference * difference, dtype=np.float64)
        ),
        "max_abs_reference": metrics["max_abs_reference"],
        "maximum_absolute_difference": metrics[
            "maximum_absolute_difference"
        ],
        "maximum_location": location,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an all-frame shard result")
    if len(args.bout_output) != 4:
        raise ValueError("all-frame forward oracle requires four rank outputs")
    manifest = verify_manifest(args)
    extraction, extraction_shard, start, stop = validate_extraction(args, manifest)

    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for shard comparison") from error

    with netCDF4.Dataset(args.canonical, "r") as canonical_file:
        dimensions = {
            name: int(len(dimension))
            for name, dimension in canonical_file.dimensions.items()
        }
        if dimensions != {"frame": 78, "x": 64, "y": 32, "z": 81, "side": 2}:
            raise ValueError(f"unexpected canonical shard dimensions: {dimensions}")
        frames = np.asarray(canonical_file.variables["frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, np.arange(start, stop, dtype=np.int64)):
            raise ValueError("canonical shard frames differ")
        canonical = {
            field: np.asarray(canonical_file.variables[field][:], dtype=np.float64)
            for field in INPUT_FIELDS
        }
        saved_midpoint = np.asarray(
            canonical_file.variables["saved_midpoint"][:], dtype=np.float64
        )
    if any(values.shape != EXPECTED_SHAPE for values in canonical.values()):
        raise ValueError("canonical field shape differs")
    if saved_midpoint.shape != (78, 2, 32):
        raise ValueError("canonical saved-midpoint shape differs")
    if any(not np.all(np.isfinite(values)) for values in canonical.values()):
        raise ValueError("canonical field is non-finite")

    raw_negative_pe = int(np.count_nonzero(canonical["Pe"] < 0.0))
    raw_negative_pi = int(np.count_nonzero(canonical["Pi"] < 0.0))
    if raw_negative_pe != 0:
        raise ValueError("canonical shard contains unexpected negative raw Pe")
    if raw_negative_pi != manifest["raw_pressure_identity"][
        "negative_raw_Pi_count_by_shard"
    ][args.shard_index]:
        raise ValueError("canonical shard negative raw Pi count differs")

    expected_runtime = {
        field: corrected.runtime_pressure(
            canonical[field], canonical["Ne"], PRESSURE_FLOOR
        )
        for field in ("Pe", "Pi")
    }
    all_cells = np.ones((64, 32), dtype=bool)
    compiled_vort = np.full(EXPECTED_SHAPE, np.nan, dtype=np.float64)
    compiled_runtime = {
        field: np.full(EXPECTED_SHAPE, np.nan, dtype=np.float64)
        for field in ("Pe", "Pi")
    }

    with netCDF4.Dataset(args.grid, "r") as grid_file:
        geometry = base.build_geometry(grid_file)
    if geometry.jacobian.shape != (64, 32):
        raise ValueError("geometry crop has unexpected shape")

    with ExitStack() as stack:
        outputs = [
            stack.enter_context(netCDF4.Dataset(path, "r"))
            for path in args.bout_output
        ]
        decomposition = corrected.rank_decomposition(outputs, list(args.bout_output))
        for output in outputs:
            if not math.isclose(
                base.scalar_float(output, "paper0_rho_s0_meters"),
                RHO_S0_METERS,
                rel_tol=1e-14,
                abs_tol=0.0,
            ):
                raise ValueError("compiled rho_s0 differs")
            for variable, expected in (
                ("paper0_zperiod", 5),
                ("paper0_forward_mode_max", 40),
                ("paper0_manufactured_mode_k", MANUFACTURED_MODE),
                ("paper0_shard_start", start),
                ("paper0_shard_stop", stop),
                ("paper0_shard_frame_count", FRAME_COUNT),
            ):
                if base.scalar_integer(output, variable) != expected:
                    raise ValueError(f"compiled {variable} differs")
            for variable, expected in (
                ("paper0_pressure_density_floor", PRESSURE_FLOOR),
                ("paper0_pressure_correction_denominator", PRESSURE_DENOMINATOR),
                ("paper0_constant_null_value", CONSTANT_NULL_VALUE),
                ("paper0_gauge_shift", GAUGE_SHIFT),
            ):
                if base.scalar_float(output, variable) != expected:
                    raise ValueError(f"compiled {variable} differs")

        for position, frame in enumerate(range(start, stop)):
            label = base.frame_label(frame)
            for output in outputs:
                if base.scalar_integer(output, f"canonical_frame_index_{label}") != frame:
                    raise ValueError(f"compiled frame marker differs for {label}")
            compiled_vort[position] = base.assembled_xyz(
                outputs, decomposition, f"input_Vort_{label}"
            )[base.MODEL_X_SLICE]
            for field in ("Pe", "Pi"):
                compiled_runtime[field][position] = base.assembled_xyz(
                    outputs, decomposition, f"runtime_{field}_{label}"
                )[base.MODEL_X_SLICE]

        constant_forward = base.assembled_xyz(
            outputs, decomposition, "constant_forward_vort"
        )[base.MODEL_X_SLICE]
        gauge_base = base.assembled_xyz(
            outputs, decomposition, "gauge_forward_base"
        )[base.MODEL_X_SLICE]
        gauge_shifted = base.assembled_xyz(
            outputs, decomposition, "gauge_forward_shifted"
        )[base.MODEL_X_SLICE]
        manufactured = base.assembled_xyz(
            outputs, decomposition, "manufactured_u"
        )[base.MODEL_X_SLICE]
        manufactured_forward = base.assembled_xyz(
            outputs, decomposition, "manufactured_forward_vort"
        )[base.MODEL_X_SLICE]
        manufactured_reconstruction = base.assembled_xyz(
            outputs, decomposition, "manufactured_reconstructed_u"
        )[base.MODEL_X_SLICE]

        input_per_frame = {
            base.frame_label(frame): base.bitwise_metrics(
                compiled_vort[position], canonical["Vort"][position]
            )
            for position, frame in enumerate(range(start, stop))
        }
        input_gate_passed = bool(
            all(metrics["passed"] for metrics in input_per_frame.values())
        )

        pressure_atol = float(manifest["runtime_pressure"]["atol"])
        pressure_rtol = float(manifest["runtime_pressure"]["rtol"])
        pressure_per_frame = {
            base.frame_label(frame): {
                field: base.gate_metrics(
                    compiled_runtime[field][position],
                    expected_runtime[field][position],
                    all_cells,
                    location_axes=("x", "y", "z"),
                    atol=pressure_atol,
                    rtol=pressure_rtol,
                )
                for field in ("Pe", "Pi")
            }
            for position, frame in enumerate(range(start, stop))
        }
        pressure_gate_passed = bool(
            all(
                metrics["passed"]
                for frame_metrics in pressure_per_frame.values()
                for metrics in frame_metrics.values()
            )
        )

        constant_metrics = base.gate_metrics(
            constant_forward,
            np.zeros_like(constant_forward),
            all_cells,
            location_axes=("x", "y", "z"),
            atol=args.atol,
            rtol=args.rtol,
        )
        gauge_metrics = base.gate_metrics(
            gauge_shifted,
            gauge_base,
            all_cells,
            location_axes=("x", "y", "z"),
            atol=args.atol,
            rtol=args.rtol,
        )
        manufactured_metrics = base.gate_metrics(
            manufactured_reconstruction,
            manufactured,
            all_cells,
            location_axes=("x", "y", "z"),
            atol=args.atol,
            rtol=args.rtol,
        )
        manufactured_fft = np.fft.rfft(manufactured, axis=-1)
        manufactured_amplitude = np.mean(
            np.abs(manufactured_fft), axis=(0, 1), dtype=np.float64
        )
        manufactured_modes_present = bool(
            manufactured_amplitude[0] > 1e-12
            and manufactured_amplitude[MANUFACTURED_MODE] > 1e-12
        )
        compiled_gate_passed = bool(
            constant_metrics["passed"]
            and gauge_metrics["passed"]
            and manufactured_metrics["passed"]
            and manufactured_modes_present
            and np.all(np.isfinite(manufactured_forward))
        )

        preliminary_passed = bool(
            input_gate_passed and pressure_gate_passed and compiled_gate_passed
        )
        forward_vort: np.ndarray | None = None
        if preliminary_passed:
            forward_vort = np.full(EXPECTED_SHAPE, np.nan, dtype=np.float64)
            for position, frame in enumerate(range(start, stop)):
                label = base.frame_label(frame)
                forward_vort[position] = base.assembled_xyz(
                    outputs, decomposition, f"forward_Vort_{label}"
                )[base.MODEL_X_SLICE]

    runtime_gate = {
        "atol": pressure_atol,
        "rtol": pressure_rtol,
        "per_frame": pressure_per_frame,
        "negative_raw_Pe_count": raw_negative_pe,
        "negative_raw_Pi_count": raw_negative_pi,
        "all_runtime_field_gates_passed": pressure_gate_passed,
        "passed": pressure_gate_passed,
    }
    compiled_gate = {
        "constant_null": constant_metrics,
        "gauge_invariance": gauge_metrics,
        "manufactured_forward_inverse_round_trip": manufactured_metrics,
        "manufactured_mode_mean_amplitude": {
            "k0": float(manufactured_amplitude[0]),
            "k3": float(manufactured_amplitude[MANUFACTURED_MODE]),
        },
        "manufactured_modes_k0_k3_present": manufactured_modes_present,
        "manufactured_forward_nonfinite_count": int(
            np.count_nonzero(~np.isfinite(manufactured_forward))
        ),
        "passed": compiled_gate_passed,
    }

    if forward_vort is None:
        source_gate: dict[str, Any] = {
            "status": "blocked_by_preliminary_gate",
            "passed": False,
            "per_frame": None,
            "pooled": None,
            "by_geometry_region_pooled": None,
            "toroidal_mode_residual": None,
            "merge_sufficient_statistics": None,
        }
    else:
        per_frame = {
            base.frame_label(frame): base.gate_metrics(
                forward_vort[position],
                canonical["Vort"][position],
                all_cells,
                location_axes=("x", "y", "z"),
                atol=args.atol,
                rtol=args.rtol,
            )
            for position, frame in enumerate(range(start, stop))
        }
        pooled = base.continuous_metrics(
            forward_vort,
            canonical["Vort"],
            all_cells,
            location_axes=("shard_frame_position", "x", "y", "z"),
        )
        if pooled["maximum_location"] is not None:
            pooled["maximum_location"]["frame_index"] = start + pooled[
                "maximum_location"
            ].pop("shard_frame_position")
        region_masks = base.potential_region_masks(geometry.region_masks)
        regional = {
            name: base.continuous_metrics(
                forward_vort,
                canonical["Vort"],
                mask,
                location_axes=("shard_frame_position", "x", "y", "z"),
            )
            for name, mask in region_masks.items()
        }
        for metrics in regional.values():
            if metrics["maximum_location"] is not None:
                metrics["maximum_location"]["frame_index"] = start + metrics[
                    "maximum_location"
                ].pop("shard_frame_position")
        mode_per_frame = {
            base.frame_label(frame): selected.mode_residual_summary(
                forward_vort[position], canonical["Vort"][position], zperiod=5
            )
            for position, frame in enumerate(range(start, stop))
        }
        mode_pooled = selected.mode_residual_summary(
            forward_vort, canonical["Vort"], zperiod=5
        )
        sufficient = {
            "full_domain": sufficient_statistics(
                forward_vort,
                canonical["Vort"],
                all_cells,
                frame_start=start,
                location_axes=("shard_frame_position", "x", "y", "z"),
            ),
            "regions": {
                name: sufficient_statistics(
                    forward_vort,
                    canonical["Vort"],
                    mask,
                    frame_start=start,
                    location_axes=("shard_frame_position", "x", "y", "z"),
                )
                for name, mask in region_masks.items()
            },
            "mode_reference_power": mode_pooled["reference_power"],
            "mode_residual_power": mode_pooled["residual_power"],
        }
        source_passed = bool(
            all(metrics["passed"] for metrics in per_frame.values())
        )
        source_gate = {
            "status": "evaluated_after_preliminary_gates_passed",
            "atol": args.atol,
            "rtol": args.rtol,
            "scope": "all_78x64x32x81_physical_points_in_shard",
            "per_frame": per_frame,
            "pooled": pooled,
            "by_geometry_region_pooled": regional,
            "toroidal_mode_residual": {
                "per_frame": mode_per_frame,
                "pooled": mode_pooled,
            },
            "merge_sufficient_statistics": sufficient,
            "passed": source_passed,
        }

    final_passed = bool(source_gate["passed"])
    result = {
        "schema_version": 1,
        "phase": "phase2_potential_vorticity_all_frame_85604_shard",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "shard_index": args.shard_index,
        "start": start,
        "stop": stop,
        "frame_indices": list(range(start, stop)),
        "physical_shape_xyz": [64, 32, 81],
        "native_z_samples": 81,
        "zperiod": 5,
        "extraction_gate": {
            "processor_coverage_complete": extraction[
                "processor_coverage_complete"
            ],
            "boundary_y_coverage_complete": extraction[
                "boundary_y_coverage_complete"
            ],
            "rank_files_traversed_once": extraction["rank_files_traversed_once"],
            "canonical_array_sha256": extraction_shard["array_sha256"],
            "passed": True,
        },
        "input_vorticity_echo_gate": {
            "per_frame": input_per_frame,
            "passed": input_gate_passed,
        },
        "runtime_pressure_gate": runtime_gate,
        "compiled_known_answer_gate": compiled_gate,
        "source_forward_closure_gate": source_gate,
        "decision": {
            "shard_passed": final_passed,
            "establishes_all_frame_closure": False,
            "automatic_state_change_authorized": False,
            "automatic_training_authorized": False,
            "automatic_held_out_access_authorized": False,
        },
        "artifacts": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "protocol": str(args.protocol),
            "protocol_sha256": sha256_file(args.protocol),
            "canonical": str(args.canonical),
            "canonical_sha256": sha256_file(args.canonical),
            "extraction_record": str(args.extraction_record),
            "extraction_record_sha256": sha256_file(args.extraction_record),
            "grid": str(args.grid),
            "grid_sha256": sha256_file(args.grid),
            "bout_outputs": [str(path) for path in args.bout_output],
            "bout_output_sha256": {
                str(path): sha256_file(path) for path in args.bout_output
            },
            "comparator": str(Path(__file__).resolve()),
            "comparator_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_strict_json(args.output, result)
    print(
        json.dumps(
            {
                "shard_index": args.shard_index,
                "start": start,
                "stop": stop,
                "input_gate_passed": input_gate_passed,
                "runtime_pressure_gate_passed": pressure_gate_passed,
                "compiled_known_answer_gate_passed": compiled_gate_passed,
                "source_forward_closure_gate_passed": final_passed,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
