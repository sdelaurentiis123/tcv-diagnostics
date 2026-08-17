#!/usr/bin/env python3
"""Validate Hermes runtime pressure, then run the unchanged paired comparator."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compare_potential_elliptic_oracle as base  # noqa: E402


BASE_COMPARATOR = TOOLS / "compare_potential_elliptic_oracle.py"
PRESSURE_FIELDS = ("Pe", "Pi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction-manifest", type=Path, required=True)
    parser.add_argument("--correction-protocol", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--extraction-record", type=Path, required=True)
    parser.add_argument("--bout-output", type=Path, nargs="+", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
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


def verify_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.correction_manifest)
    if (
        manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
        or manifest["training_allowed"]
    ):
        raise ValueError("correction manifest violates development-only scope")
    for name, lock in manifest["predecessor_locks"].items():
        path = ROOT / lock["path"]
        if base.sha256_file(path) != lock["sha256"]:
            raise ValueError(f"predecessor lock differs for {name}")
    base_lock = manifest["predecessor_locks"]["base_manifest"]
    if args.base_manifest.resolve() != (ROOT / base_lock["path"]).resolve():
        raise ValueError("base manifest path differs from correction lock")
    if base.sha256_file(args.base_manifest) != base_lock["sha256"]:
        raise ValueError("base manifest hash differs from correction lock")
    reuse = manifest["canonical_reuse"]
    if args.canonical.resolve() != Path(reuse["canonical_path"]).resolve():
        raise ValueError("canonical path differs from correction lock")
    if base.sha256_file(args.canonical) != reuse["canonical_sha256"]:
        raise ValueError("canonical hash differs from correction lock")
    if args.extraction_record.resolve() != Path(
        reuse["extraction_record_path"]
    ).resolve():
        raise ValueError("extraction-record path differs from correction lock")
    if (
        base.sha256_file(args.extraction_record)
        != reuse["extraction_record_sha256"]
    ):
        raise ValueError("extraction-record hash differs from correction lock")
    if args.atol != manifest["source_reconstruction_gate"]["continuous_atol"]:
        raise ValueError("source reconstruction atol differs from correction lock")
    if args.rtol != manifest["source_reconstruction_gate"]["continuous_rtol"]:
        raise ValueError("source reconstruction rtol differs from correction lock")
    return manifest


def runtime_pressure(
    evolved_pressure: np.ndarray,
    density: np.ndarray,
    density_floor: float,
) -> np.ndarray:
    """Independent array form of EvolvePressure::transform_impl."""

    evolved = np.asarray(evolved_pressure, dtype=np.float64)
    number_density = np.asarray(density, dtype=np.float64)
    if evolved.shape != number_density.shape:
        raise ValueError("pressure and density shapes differ")
    if not np.isfinite(density_floor) or density_floor <= 0.0:
        raise ValueError("density floor must be positive and finite")
    nonnegative_density = np.maximum(number_density, 0.0)
    soft_density = nonnegative_density + density_floor * np.exp(
        -nonnegative_density / density_floor
    )
    temperature = np.maximum(evolved, 0.0) / soft_density
    result = number_density * temperature
    if not np.all(np.isfinite(result)):
        raise ValueError("independent runtime pressure is non-finite")
    return result


def rank_decomposition(outputs: list[Any], paths: list[Path]) -> list[dict[str, Any]]:
    decomposition = []
    for output, path in zip(outputs, paths):
        metadata = {
            "path": str(path),
            "NXPE": base.scalar_integer(output, "NXPE"),
            "NYPE": base.scalar_integer(output, "NYPE"),
            "MYSUB": base.scalar_integer(output, "MYSUB"),
            "PE_XIND": base.scalar_integer(output, "PE_XIND"),
            "PE_YIND": base.scalar_integer(output, "PE_YIND"),
        }
        if (
            metadata["NXPE"] != 1
            or metadata["NYPE"] != 4
            or metadata["MYSUB"] != 8
            or metadata["PE_XIND"] != 0
        ):
            raise ValueError(f"unexpected rank decomposition: {metadata}")
        if base.scalar_integer(output, "paper0_runtime_pressure_correction") != 1:
            raise ValueError("compiled output did not enable runtime pressure")
        decomposition.append(metadata)
    if sorted(item["PE_YIND"] for item in decomposition) != [0, 1, 2, 3]:
        raise ValueError("rank outputs do not cover four unique y partitions")
    return decomposition


def validate_runtime_pressure(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for the corrected oracle") from error

    with netCDF4.Dataset(args.canonical, "r") as canonical_file:
        canonical = {
            name: np.asarray(canonical_file.variables[name][:], dtype=np.float64)
            for name in ("Ne", "Pe", "Pi")
        }
        frames = np.asarray(
            canonical_file.variables["frame_index"][:], dtype=np.int64
        )
    if not np.array_equal(frames, base.FRAME_INDICES):
        raise ValueError("canonical frame indices differ from correction protocol")

    density_floor = float(
        manifest["source_contract_correction"]["density_floor"]
    )
    expected = {
        field: runtime_pressure(canonical[field], canonical["Ne"], density_floor)
        for field in PRESSURE_FIELDS
    }
    expected["Pi_hat"] = expected["Pi"] - expected["Pe"] / base.PRESSURE_DENOMINATOR
    actual = {
        field: np.full_like(canonical["Ne"], np.nan)
        for field in ("Pe", "Pi", "Pi_hat")
    }

    with ExitStack() as stack:
        outputs = [
            stack.enter_context(netCDF4.Dataset(path, "r"))
            for path in args.bout_output
        ]
        decomposition = rank_decomposition(outputs, list(args.bout_output))
        for output in outputs:
            compiled_floor = base.scalar_float(
                output, "paper0_pressure_density_floor"
            )
            if compiled_floor != density_floor:
                raise ValueError("compiled density floor differs from correction lock")
        for position, frame in enumerate(base.FRAME_INDICES):
            label = base.frame_label(frame)
            names = {
                "Pe": f"runtime_Pe_{label}",
                "Pi": f"runtime_Pi_{label}",
                "Pi_hat": f"pi_hat_{label}",
            }
            for field, variable in names.items():
                actual[field][position] = base.assembled_xyz(
                    outputs, decomposition, variable
                )[base.MODEL_X_SLICE]

    all_cells = np.ones((64, 32), dtype=bool)
    gate = manifest["runtime_pressure_gate"]
    atol = float(gate["continuous_atol"])
    rtol = float(gate["continuous_rtol"])
    per_frame: dict[str, Any] = {}
    for position, frame in enumerate(base.FRAME_INDICES):
        label = base.frame_label(frame)
        per_frame[label] = {
            field: base.gate_metrics(
                actual[field][position],
                expected[field][position],
                all_cells,
                location_axes=("x", "y", "z"),
                atol=atol,
                rtol=rtol,
            )
            for field in ("Pe", "Pi", "Pi_hat")
        }

    negative_pe = np.argwhere(canonical["Pe"] < 0.0)
    negative_pi = np.argwhere(canonical["Pi"] < 0.0)
    known = manifest["known_failed_point"]
    expected_position = list(base.FRAME_INDICES).index(known["frame_index"])
    expected_index = np.asarray(
        [expected_position, *known["model_indices_xyz"]], dtype=np.int64
    )
    known_support_passed = bool(
        negative_pe.shape == (known["selected_frame_negative_raw_Pe_count"], 4)
        and negative_pi.shape == (known["selected_frame_negative_raw_Pi_count"], 4)
        and negative_pi.shape[0] == 1
        and np.array_equal(negative_pi[0], expected_index)
    )
    known_runtime_pi = float(actual["Pi"][tuple(expected_index)])
    known_runtime_zero_passed = bool(known_runtime_pi == 0.0)
    field_gates_passed = all(
        metrics["passed"]
        for frame_metrics in per_frame.values()
        for metrics in frame_metrics.values()
    )
    passed = bool(
        field_gates_passed
        and known_support_passed
        and known_runtime_zero_passed
    )
    return {
        "continuous_atol": atol,
        "continuous_rtol": rtol,
        "per_frame": per_frame,
        "selected_frame_negative_raw_Pe_count": int(negative_pe.shape[0]),
        "selected_frame_negative_raw_Pi_count": int(negative_pi.shape[0]),
        "known_negative_raw_Pi_location": {
            "selected_frame_position": int(expected_index[0]),
            "frame_index": int(known["frame_index"]),
            "x": int(expected_index[1]),
            "y": int(expected_index[2]),
            "z": int(expected_index[3]),
        },
        "known_negative_raw_Pi": float(canonical["Pi"][tuple(expected_index)]),
        "compiled_runtime_Pi_at_known_point": known_runtime_pi,
        "known_negative_support_passed": known_support_passed,
        "known_runtime_Pi_zero_passed": known_runtime_zero_passed,
        "all_runtime_field_gates_passed": field_gates_passed,
        "passed": passed,
    }


def common_artifacts(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    failed = manifest["predecessor_locks"]["failed_replay_result"]
    return {
        "runtime_pressure_correction_manifest": str(args.correction_manifest),
        "runtime_pressure_correction_manifest_sha256": base.sha256_file(
            args.correction_manifest
        ),
        "runtime_pressure_correction_protocol": str(args.correction_protocol),
        "runtime_pressure_correction_protocol_sha256": base.sha256_file(
            args.correction_protocol
        ),
        "predecessor_failed_result": str(ROOT / failed["path"]),
        "predecessor_failed_result_sha256": failed["sha256"],
        "base_comparator": str(BASE_COMPARATOR),
        "base_comparator_sha256": base.sha256_file(BASE_COMPARATOR),
    }


def write_runtime_failure(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    runtime_gate: dict[str, Any],
) -> int:
    result = {
        "schema_version": 2,
        "phase": "phase2_potential_elliptic_85604_runtime_pressure_correction",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "runtime_pressure_transformation_gate": runtime_gate,
        "source_reconstruction_gate": {
            "status": "not_evaluated_after_runtime_pressure_gate_failure",
            "passed": False,
        },
        "paired_boundary_effect": {
            "status": "blocked_by_runtime_pressure_transformation_gate",
            "materiality_label_assigned": False,
            "potential": None,
            "transport": None,
        },
        "decision": {
            "paired_effect_interpretable": False,
            "automatic_state_change_authorized": False,
            "automatic_training_authorized": False,
            "automatic_held_out_access_authorized": False,
        },
        "artifacts": {
            **common_artifacts(args, manifest),
            "canonical": str(args.canonical),
            "canonical_sha256": base.sha256_file(args.canonical),
            "bout_outputs": [str(path) for path in args.bout_output],
            "bout_output_sha256": {
                str(path): base.sha256_file(path) for path in args.bout_output
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 1


def run_base_comparator(args: argparse.Namespace) -> tuple[list[str], int]:
    command = [
        sys.executable,
        str(BASE_COMPARATOR),
        "--manifest",
        str(args.base_manifest),
        "--canonical",
        str(args.canonical),
        "--extraction-record",
        str(args.extraction_record),
        "--bout-output",
        *[str(path) for path in args.bout_output],
        "--grid",
        str(args.grid),
        "--output",
        str(args.base_output),
        "--arrays",
        str(args.arrays),
        "--paper0-commit",
        args.paper0_commit,
        "--slurm-job-id",
        str(args.slurm_job_id),
        "--atol",
        repr(args.atol),
        "--rtol",
        repr(args.rtol),
    ]
    completed = subprocess.run(command, check=False)
    return command, int(completed.returncode)


def finalize_base_result(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    runtime_gate: dict[str, Any],
    command: list[str],
    base_status: int,
) -> int:
    if not args.base_output.exists():
        raise FileNotFoundError("base comparator did not write its strict result")
    original = load_json(args.base_output)
    if original["paper0_commit"] != args.paper0_commit:
        raise ValueError("base comparison commit differs from corrected job")
    if original["slurm_job_id"] != args.slurm_job_id:
        raise ValueError("base comparison Slurm job differs from corrected job")
    if original["held_out_85606_read"] or original["training_performed"]:
        raise ValueError("base comparison violates corrected scope")
    base_gate_passed = bool(original["source_reconstruction_gate"]["passed"])
    if (base_status == 0) != base_gate_passed:
        raise ValueError("base comparator exit status disagrees with its gate")

    result = copy.deepcopy(original)
    result["schema_version"] = 2
    result["phase"] = (
        "phase2_potential_elliptic_85604_runtime_pressure_correction"
    )
    result["equation"] = {
        "raw_evolved_pressure_fields": ["Pe", "Pi"],
        "runtime_pressure": (
            "P_runtime = N * max(P_evolved,0) / softFloor(N,1e-7)"
        ),
        "soft_floor": (
            "softFloor(N,f) = max(N,0) + f * exp(-max(N,0)/f)"
        ),
        "pressure_correction": (
            "Pi_hat = Pi_runtime - Pe_runtime / 3672"
        ),
        "coefficient_C": "2 / Bxy^2",
        "right_hand_side": "Vort * Bxy^2 / 2",
        "solver_type": "cyclic",
        "split_n0": False,
        "radial_boundary_flags": "INVERT_SET",
    }
    result["runtime_pressure_transformation_gate"] = runtime_gate
    corrected_gate_passed = bool(runtime_gate["passed"] and base_gate_passed)
    result["source_reconstruction_gate"]["passed"] = corrected_gate_passed
    result["source_reconstruction_gate"][
        "requires_runtime_pressure_transformation_gate"
    ] = True
    if not corrected_gate_passed:
        result["paired_boundary_effect"] = {
            "status": "blocked_by_corrected_source_reconstruction_gate",
            "materiality_label_assigned": False,
            "potential": None,
            "transport": None,
        }
    result["decision"]["paired_effect_interpretable"] = corrected_gate_passed
    result["decision"]["automatic_held_out_access_authorized"] = False
    base_manifest_artifact = {
        "path": result["artifacts"].pop("manifest"),
        "sha256": result["artifacts"].pop("manifest_sha256"),
        "role": "canonical extraction and unchanged base comparison contract",
    }
    result["artifacts"].update(common_artifacts(args, manifest))
    result["artifacts"]["base_manifest"] = base_manifest_artifact
    result["artifacts"]["base_comparison"] = str(args.base_output)
    result["artifacts"]["base_comparison_sha256"] = base.sha256_file(
        args.base_output
    )
    result["artifacts"]["base_comparator_command"] = command
    result["artifacts"]["base_comparator_exit_status"] = base_status

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "runtime_pressure_gate_passed": runtime_gate["passed"],
                "source_reconstruction_gate_passed": corrected_gate_passed,
                "paired_boundary_effect_status": result[
                    "paired_boundary_effect"
                ]["status"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if corrected_gate_passed else 1


def main() -> int:
    args = parse_args()
    for path in (args.base_output, args.output, args.arrays):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if len(args.bout_output) != 4:
        raise ValueError("corrected oracle requires four rank outputs")
    manifest = verify_manifest(args)
    runtime_gate = validate_runtime_pressure(args, manifest)
    if not runtime_gate["passed"]:
        return write_runtime_failure(args, manifest, runtime_gate)
    command, base_status = run_base_comparator(args)
    return finalize_base_result(
        args,
        manifest,
        runtime_gate,
        command,
        base_status,
    )


if __name__ == "__main__":
    raise SystemExit(main())
