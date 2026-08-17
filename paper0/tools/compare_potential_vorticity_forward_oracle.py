#!/usr/bin/env python3
"""Validate the cyclic forward implementation, then compare it with stored Vort."""

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


FRAME_INDICES = (0, 156, 312, 467, 623)
INPUT_FIELDS = ("Ne", "Pe", "Pi", "Vort", "phi")
BOUNDARY_SIDES = ("inner", "outer")
EXPECTED_SHAPE = (5, 64, 32, 81)
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
    parser.add_argument("--bout-output", type=Path, nargs="+", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
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


def resolve_tracked(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def require_lock(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 differs: {actual} != {expected}")


def verify_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    if manifest["phase"] != "phase2_potential_vorticity_forward_85604":
        raise ValueError("unexpected forward manifest phase")
    scope = manifest["scope"]
    if (
        scope["development_run"] != "85604"
        or scope["held_out_85606_access_allowed"]
        or scope["training_allowed"]
        or scope["model_evaluation"]
        or scope["codec_evaluation"]
        or scope["assimilation_evaluation"]
    ):
        raise ValueError("forward manifest violates development-only scope")
    if scope["frame_indices"] != list(FRAME_INDICES):
        raise ValueError("forward manifest frame set differs")
    if scope["physical_shape_xyz"] != [64, 32, 81]:
        raise ValueError("forward manifest physical shape differs")
    if scope["zperiod"] != 5 or scope["toroidal_mode_mapping"] != "n=5k":
        raise ValueError("forward manifest toroidal mapping differs")

    protocol_lock = manifest["protocol"]
    if args.protocol.resolve() != (ROOT / protocol_lock["path"]).resolve():
        raise ValueError("forward protocol path differs from manifest")
    require_lock(args.protocol, protocol_lock["sha256"], "forward protocol")

    inputs = manifest["immutable_inputs"]
    for argument, key in (
        (args.canonical, "canonical"),
        (args.extraction_record, "extraction_record"),
        (args.grid, "grid"),
    ):
        lock = inputs[key]
        if argument.resolve() != Path(lock["path"]).resolve():
            raise ValueError(f"{key} path differs from manifest")
        require_lock(argument, lock["sha256"], key)
    for key in (
        "accepted_inverse_result",
        "runtime_pressure_correction_manifest",
        "runtime_pressure_correction_protocol",
        "bout_input",
    ):
        lock = inputs[key]
        require_lock(resolve_tracked(lock["path"]), lock["sha256"], key)

    source_gate = manifest["ordered_gates"]["source_forward_closure"]
    if args.atol != float(source_gate["atol"]):
        raise ValueError("forward atol differs from frozen manifest")
    if args.rtol != float(source_gate["rtol"]):
        raise ValueError("forward rtol differs from frozen manifest")
    return manifest


def apply_complex_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Apply a complex tridiagonal matrix with explicit end coefficients."""

    arrays = [np.asarray(item, dtype=np.complex128) for item in (
        lower,
        diagonal,
        upper,
        values,
    )]
    if any(item.ndim != 1 for item in arrays):
        raise ValueError("tridiagonal inputs must be one-dimensional")
    if len({item.shape for item in arrays}) != 1:
        raise ValueError("tridiagonal inputs must have identical shape")
    lower_array, diagonal_array, upper_array, value_array = arrays
    result = diagonal_array * value_array
    if result.size:
        result[1:] += lower_array[1:] * value_array[:-1]
        result[:-1] += upper_array[:-1] * value_array[1:]
    if not np.all(np.isfinite(result)):
        raise ValueError("tridiagonal application produced non-finite values")
    return result


def mode_residual_summary(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    zperiod: int,
) -> dict[str, Any]:
    """Return mode-wise reference and residual power on the last axis."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != reference_array.shape:
        raise ValueError("mode-summary candidate and reference shapes differ")
    if candidate_array.shape[-1] != 81:
        raise ValueError("mode summary requires the native 81-cell toroidal axis")
    if zperiod != 5:
        raise ValueError("forward mode summary requires zperiod=5")
    if not (
        np.all(np.isfinite(candidate_array))
        and np.all(np.isfinite(reference_array))
    ):
        raise ValueError("mode summary inputs must be finite")
    reference_fft = np.fft.rfft(reference_array, axis=-1)
    residual_fft = np.fft.rfft(candidate_array - reference_array, axis=-1)
    reduction_axes = tuple(range(reference_fft.ndim - 1))
    reference_power = np.sum(
        np.abs(reference_fft) ** 2, axis=reduction_axes, dtype=np.float64
    )
    residual_power = np.sum(
        np.abs(residual_fft) ** 2, axis=reduction_axes, dtype=np.float64
    )
    reference_power_floor = float(
        np.finfo(np.float64).eps
        * max(float(np.max(reference_power, initial=0.0)), 1.0)
    )
    relative = np.divide(
        residual_power,
        reference_power,
        out=np.full_like(residual_power, np.nan),
        where=reference_power > reference_power_floor,
    )
    return {
        "fourier_index_k": list(range(reference_power.size)),
        "toroidal_mode_n": [zperiod * k for k in range(reference_power.size)],
        "reference_power": reference_power.tolist(),
        "residual_power": residual_power.tolist(),
        "relative_power_denominator_floor": reference_power_floor,
        "relative_residual_power": [
            None if not np.isfinite(value) else float(value) for value in relative
        ],
    }


def assemble_boundary(
    outputs: list[Any],
    decomposition: list[dict[str, Any]],
    variable: str,
) -> np.ndarray:
    return base.assemble_xy_partitions(
        [
            (
                metadata["PE_YIND"],
                base.canonical_xy(output.variables[variable]),
            )
            for metadata, output in zip(decomposition, outputs, strict=True)
        ]
    )[base.MODEL_X_SLICE]


def validate_extraction(path: Path, canonical: Path) -> dict[str, Any]:
    extraction = load_json(path)
    if (
        extraction["development_run"] != "85604"
        or extraction["held_out_run_read"]
        or extraction["training_performed"]
    ):
        raise ValueError("canonical extraction violates forward scope")
    if extraction["frame_indices"] != list(FRAME_INDICES):
        raise ValueError("canonical extraction frame set differs")
    if extraction["canonical_file_sha256"] != sha256_file(canonical):
        raise ValueError("canonical extraction digest differs")
    return extraction


def write_strict_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.arrays.exists():
        raise FileExistsError("refusing to overwrite forward comparison artifacts")
    if len(args.bout_output) != 4:
        raise ValueError("forward oracle requires exactly four rank outputs")
    manifest = verify_manifest(args)
    extraction = validate_extraction(args.extraction_record, args.canonical)

    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for the forward oracle") from error

    with netCDF4.Dataset(args.canonical, "r") as canonical_file:
        dimensions = {
            name: int(len(dimension))
            for name, dimension in canonical_file.dimensions.items()
        }
        if dimensions != {
            "selected_frame": 5,
            "x": 64,
            "y": 32,
            "z": 81,
            "side": 2,
        }:
            raise ValueError(f"unexpected canonical dimensions: {dimensions}")
        frames = np.asarray(
            canonical_file.variables["frame_index"][:], dtype=np.int64
        )
        if not np.array_equal(frames, FRAME_INDICES):
            raise ValueError("canonical frames differ from forward protocol")
        canonical = {
            field: np.asarray(
                canonical_file.variables[field][:], dtype=np.float64
            )
            for field in INPUT_FIELDS
        }
        saved_midpoint = np.asarray(
            canonical_file.variables["saved_midpoint"][:], dtype=np.float64
        )
    if any(values.shape != EXPECTED_SHAPE for values in canonical.values()):
        raise ValueError("canonical field shape differs from forward protocol")
    if saved_midpoint.shape != (5, 2, 32):
        raise ValueError("canonical saved midpoint shape differs")

    expected_runtime = {
        field: corrected.runtime_pressure(
            canonical[field], canonical["Ne"], PRESSURE_FLOOR
        )
        for field in ("Pe", "Pi")
    }
    expected_runtime["Pi_hat"] = (
        expected_runtime["Pi"]
        - expected_runtime["Pe"] / PRESSURE_DENOMINATOR
    )
    all_cells = np.ones((64, 32), dtype=bool)
    pressure_actual = {
        field: np.full(EXPECTED_SHAPE, np.nan, dtype=np.float64)
        for field in ("Pe", "Pi", "Pi_hat")
    }
    forward_u = np.full(EXPECTED_SHAPE, np.nan, dtype=np.float64)
    input_echoes: dict[str, Any] = {}
    boundary_echoes: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}

    with netCDF4.Dataset(args.grid, "r") as grid_file:
        geometry = base.build_geometry(grid_file)
    if geometry.jacobian.shape != (64, 32):
        raise ValueError("forward geometry crop has unexpected shape")

    with ExitStack() as stack:
        outputs = [
            stack.enter_context(netCDF4.Dataset(path, "r"))
            for path in args.bout_output
        ]
        decomposition = corrected.rank_decomposition(
            outputs, list(args.bout_output)
        )
        for output in outputs:
            if not math.isclose(
                base.scalar_float(output, "paper0_rho_s0_meters"),
                RHO_S0_METERS,
                rel_tol=1e-14,
                abs_tol=0.0,
            ):
                raise ValueError("compiled rho_s0 differs")
            if base.scalar_integer(output, "paper0_zperiod") != 5:
                raise ValueError("compiled zperiod differs")
            if base.scalar_integer(output, "paper0_forward_mode_max") != 40:
                raise ValueError("compiled forward mode range differs")
            if (
                base.scalar_integer(output, "paper0_manufactured_mode_k")
                != MANUFACTURED_MODE
            ):
                raise ValueError("compiled manufactured mode differs")
            for variable, expected in (
                ("paper0_pressure_density_floor", PRESSURE_FLOOR),
                ("paper0_pressure_correction_denominator", PRESSURE_DENOMINATOR),
                ("paper0_constant_null_value", CONSTANT_NULL_VALUE),
                ("paper0_gauge_shift", GAUGE_SHIFT),
            ):
                if base.scalar_float(output, variable) != expected:
                    raise ValueError(f"compiled {variable} differs")

        for position, frame in enumerate(FRAME_INDICES):
            label = base.frame_label(frame)
            for output in outputs:
                if (
                    base.scalar_integer(output, f"canonical_frame_index_{label}")
                    != frame
                ):
                    raise ValueError(f"compiled frame marker differs for {label}")
            frame_echoes: dict[str, Any] = {}
            for field in INPUT_FIELDS:
                compiled = base.assembled_xyz(
                    outputs, decomposition, f"input_{field}_{label}"
                )[base.MODEL_X_SLICE]
                frame_echoes[field] = base.bitwise_metrics(
                    compiled, canonical[field][position]
                )
            input_echoes[label] = frame_echoes

            frame_boundaries: dict[str, Any] = {}
            for side_index, side in enumerate(BOUNDARY_SIDES):
                compiled = assemble_boundary(
                    outputs, decomposition, f"saved_midpoint_{side}_{label}"
                )
                expected = np.broadcast_to(
                    saved_midpoint[position, side_index][None, :], (64, 32)
                )
                frame_boundaries[f"saved_midpoint_{side}"] = base.bitwise_metrics(
                    compiled, expected
                )
            boundary_echoes[label] = frame_boundaries

            for field, variable in (
                ("Pe", f"runtime_Pe_{label}"),
                ("Pi", f"runtime_Pi_{label}"),
                ("Pi_hat", f"pi_hat_{label}"),
            ):
                pressure_actual[field][position] = base.assembled_xyz(
                    outputs, decomposition, variable
                )[base.MODEL_X_SLICE]
            forward_u[position] = base.assembled_xyz(
                outputs, decomposition, f"forward_u_{label}"
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

        volume_echoes_passed = all(
            metrics["passed"]
            for frame in input_echoes.values()
            for metrics in frame.values()
        )
        boundary_echoes_passed = all(
            metrics["passed"]
            for frame in boundary_echoes.values()
            for metrics in frame.values()
        )
        input_gate_passed = bool(volume_echoes_passed and boundary_echoes_passed)

        pressure_lock = manifest["ordered_gates"]["runtime_pressure"]
        pressure_atol = float(pressure_lock["atol"])
        pressure_rtol = float(pressure_lock["rtol"])
        pressure_per_frame = {
            base.frame_label(frame): {
                field: base.gate_metrics(
                    pressure_actual[field][position],
                    expected_runtime[field][position],
                    all_cells,
                    location_axes=("x", "y", "z"),
                    atol=pressure_atol,
                    rtol=pressure_rtol,
                )
                for field in ("Pe", "Pi", "Pi_hat")
            }
            for position, frame in enumerate(FRAME_INDICES)
        }
        negative_pe = np.argwhere(canonical["Pe"] < 0.0)
        negative_pi = np.argwhere(canonical["Pi"] < 0.0)
        known_index = np.asarray([2, 6, 31, 73], dtype=np.int64)
        negative_support_passed = bool(
            negative_pe.shape == (0, 4)
            and negative_pi.shape == (1, 4)
            and np.array_equal(negative_pi[0], known_index)
        )
        compiled_known_pi = float(pressure_actual["Pi"][tuple(known_index)])
        pressure_fields_passed = all(
            metrics["passed"]
            for frame in pressure_per_frame.values()
            for metrics in frame.values()
        )
        pressure_gate_passed = bool(
            pressure_fields_passed
            and negative_support_passed
            and compiled_known_pi == 0.0
        )
        runtime_pressure_gate = {
            "atol": pressure_atol,
            "rtol": pressure_rtol,
            "per_frame": pressure_per_frame,
            "selected_frame_negative_raw_Pe_count": int(negative_pe.shape[0]),
            "selected_frame_negative_raw_Pi_count": int(negative_pi.shape[0]),
            "known_negative_raw_Pi_location": {
                "selected_frame_position": 2,
                "frame_index": 312,
                "x": 6,
                "y": 31,
                "z": 73,
            },
            "known_negative_raw_Pi": float(canonical["Pi"][tuple(known_index)]),
            "compiled_runtime_Pi_at_known_point": compiled_known_pi,
            "known_negative_support_passed": negative_support_passed,
            "all_runtime_field_gates_passed": pressure_fields_passed,
            "passed": pressure_gate_passed,
        }

        expected_u = canonical["phi"] + expected_runtime["Pi_hat"]
        forward_u_metrics = {
            base.frame_label(frame): base.gate_metrics(
                forward_u[position],
                expected_u[position],
                all_cells,
                location_axes=("x", "y", "z"),
                atol=pressure_atol,
                rtol=pressure_rtol,
            )
            for position, frame in enumerate(FRAME_INDICES)
        }
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
        manufactured_mode_amplitude = np.mean(
            np.abs(manufactured_fft), axis=(0, 1), dtype=np.float64
        )
        manufactured_modes_present = bool(
            manufactured_mode_amplitude[0] > 1e-12
            and manufactured_mode_amplitude[MANUFACTURED_MODE] > 1e-12
        )
        manufactured_forward_finite = bool(
            np.all(np.isfinite(manufactured_forward))
        )
        implementation_gate_passed = bool(
            all(metrics["passed"] for metrics in forward_u_metrics.values())
            and constant_metrics["passed"]
            and gauge_metrics["passed"]
            and manufactured_metrics["passed"]
            and manufactured_modes_present
            and manufactured_forward_finite
        )
        implementation_gate = {
            "forward_u_construction_per_frame": forward_u_metrics,
            "constant_null": constant_metrics,
            "gauge_invariance": gauge_metrics,
            "manufactured_forward_inverse_round_trip": manufactured_metrics,
            "manufactured_mode_mean_amplitude": {
                "k0": float(manufactured_mode_amplitude[0]),
                "k3": float(manufactured_mode_amplitude[MANUFACTURED_MODE]),
            },
            "manufactured_modes_k0_k3_present": manufactured_modes_present,
            "manufactured_forward_nonfinite_count": int(
                np.count_nonzero(~np.isfinite(manufactured_forward))
            ),
            "passed": implementation_gate_passed,
        }

        preliminary_passed = bool(
            input_gate_passed
            and pressure_gate_passed
            and implementation_gate_passed
        )
        forward_all: np.ndarray | None = None
        if preliminary_passed:
            forward_all = np.full(EXPECTED_SHAPE, np.nan, dtype=np.float64)
            for position, frame in enumerate(FRAME_INDICES):
                label = base.frame_label(frame)
                forward_all[position] = base.assembled_xyz(
                    outputs, decomposition, f"forward_Vort_{label}"
                )[base.MODEL_X_SLICE]

    arrays.update(
        {
            "constant_forward_vort": constant_forward,
            "gauge_forward_base": gauge_base,
            "gauge_forward_shifted": gauge_shifted,
            "manufactured_u": manufactured,
            "manufactured_forward_vort": manufactured_forward,
            "manufactured_reconstructed_u": manufactured_reconstruction,
            "runtime_Pe": pressure_actual["Pe"],
            "runtime_Pi": pressure_actual["Pi"],
            "Pi_hat": pressure_actual["Pi_hat"],
            "forward_u": forward_u,
        }
    )

    if forward_all is None:
        source_gate: dict[str, Any] = {
            "status": "blocked_by_preliminary_gate",
            "passed": False,
            "per_frame": None,
            "pooled": None,
            "by_geometry_region_pooled": None,
            "toroidal_mode_residual": None,
        }
    else:
        per_frame = {
            base.frame_label(frame): base.gate_metrics(
                forward_all[position],
                canonical["Vort"][position],
                all_cells,
                location_axes=("x", "y", "z"),
                atol=args.atol,
                rtol=args.rtol,
            )
            for position, frame in enumerate(FRAME_INDICES)
        }
        pooled = base.gate_metrics(
            forward_all,
            canonical["Vort"],
            all_cells,
            location_axes=("selected_frame_position", "x", "y", "z"),
            atol=args.atol,
            rtol=args.rtol,
        )
        regional = {
            name: base.continuous_metrics(
                forward_all,
                canonical["Vort"],
                mask,
                location_axes=("selected_frame_position", "x", "y", "z"),
            )
            for name, mask in base.potential_region_masks(
                geometry.region_masks
            ).items()
        }
        mode_summary = {
            "per_frame": {
                base.frame_label(frame): mode_residual_summary(
                    forward_all[position], canonical["Vort"][position], zperiod=5
                )
                for position, frame in enumerate(FRAME_INDICES)
            },
            "pooled": mode_residual_summary(
                forward_all, canonical["Vort"], zperiod=5
            ),
        }
        source_passed = bool(
            all(metrics["passed"] for metrics in per_frame.values())
        )
        source_gate = {
            "status": "evaluated_after_preliminary_gates_passed",
            "atol": args.atol,
            "rtol": args.rtol,
            "scope": "all_64x32x81_physical_points_for_each_selected_frame",
            "per_frame": per_frame,
            "pooled": pooled,
            "by_geometry_region_pooled": regional,
            "toroidal_mode_residual": mode_summary,
            "passed": source_passed,
        }
        arrays["forward_Vort"] = forward_all
        arrays["stored_Vort"] = canonical["Vort"]
        arrays["forward_minus_stored_Vort"] = forward_all - canonical["Vort"]

    args.arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.arrays, **arrays)
    final_passed = bool(source_gate["passed"])
    result = {
        "schema_version": 1,
        "phase": "phase2_potential_vorticity_forward_85604",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "frame_indices": list(FRAME_INDICES),
        "physical_shape_xyz": [64, 32, 81],
        "native_z_samples": 81,
        "zperiod": 5,
        "equation": {
            "runtime_pressure": "N*max(P_raw,0)/softFloor(N,1e-7)",
            "Pi_hat": "Pi_runtime-Pe_runtime/3672",
            "C": "2/Bxy^2",
            "u": "phi+Pi_hat",
            "forward_vorticity": "C*L_C(u)",
            "discrete_operator": "BOUT++ Laplacian::tridagCoefs with rfft/irfft",
            "alternative_relax_potential_fv_operator_used": False,
        },
        "input_echo_gate": {
            "volume": input_echoes,
            "boundary": boundary_echoes,
            "volume_passed": volume_echoes_passed,
            "boundary_passed": boundary_echoes_passed,
            "passed": input_gate_passed,
        },
        "runtime_pressure_gate": runtime_pressure_gate,
        "compiled_implementation_gate": implementation_gate,
        "source_forward_closure_gate": source_gate,
        "decision": {
            "selected_frame_bidirectional_closure_validated": final_passed,
            "selected_frames_establish_all_frame_stability": False,
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
            "extraction_source_manifest_sha256": extraction["manifest_sha256"],
            "grid": str(args.grid),
            "grid_sha256": sha256_file(args.grid),
            "bout_outputs": [str(path) for path in args.bout_output],
            "bout_output_sha256": {
                str(path): sha256_file(path) for path in args.bout_output
            },
            "comparison_arrays": str(args.arrays),
            "comparison_arrays_sha256": sha256_file(args.arrays),
            "comparator": str(Path(__file__).resolve()),
            "comparator_sha256": sha256_file(Path(__file__).resolve()),
            "accepted_inverse_result_sha256": manifest["immutable_inputs"][
                "accepted_inverse_result"
            ]["sha256"],
        },
    }
    write_strict_json(args.output, result)
    print(
        json.dumps(
            {
                "input_gate_passed": input_gate_passed,
                "runtime_pressure_gate_passed": pressure_gate_passed,
                "compiled_implementation_gate_passed": implementation_gate_passed,
                "source_forward_closure_gate_passed": final_passed,
                "source_status": source_gate["status"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
