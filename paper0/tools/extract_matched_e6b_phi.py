#!/usr/bin/env python3
"""Assemble compiled matched-E6B potential output and gate truth replay."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_hermes_radial_flow_oracle import assembled_xyz  # noqa: E402
from compare_shifted_ddy_oracle import (  # noqa: E402
    MODEL_X_SLICE,
    scalar_integer,
)
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    sha256_file as sha256_path,
    write_strict_json_atomic,
)


NATIVE_SHAPE = (64, 32, 81)
DEFAULT_ATOL = 5.0e-10
DEFAULT_RTOL = 5.0e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--bout-output", type=Path, nargs=4, required=True)
    parser.add_argument("--output-phi", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--truth-layout", action="store_true")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    return parser.parse_args()


def frame_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    truth = np.asarray(reference, dtype=np.float64)
    derived = np.asarray(candidate, dtype=np.float64)
    if truth.shape != NATIVE_SHAPE or derived.shape != NATIVE_SHAPE:
        raise ValueError("truth-replay arrays must have native shape")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(derived)):
        raise ValueError("truth-replay arrays contain non-finite values")
    difference = derived - truth
    maximum = float(np.max(np.abs(difference)))
    reference_maximum = float(np.max(np.abs(truth)))
    tolerance = float(atol + rtol * reference_maximum)
    denominator = float(np.sum(truth * truth, dtype=np.float64))
    numerator = float(np.sum(difference * difference, dtype=np.float64))
    return {
        "point_count": int(truth.size),
        "maximum_absolute_difference": maximum,
        "maximum_absolute_reference": reference_maximum,
        "acceptance_tolerance": tolerance,
        "rmse": math.sqrt(numerator / truth.size),
        "relative_l2": math.sqrt(numerator / denominator),
        "bias": float(np.mean(difference, dtype=np.float64)),
        "passes": maximum <= tolerance,
    }


def _write_phi(
    path: Path,
    *,
    frame_indices: np.ndarray,
    values: list[np.ndarray],
    source_input_sha256: str,
    truth_layout: bool,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["development_run"] = "85604"
            handle.attrs["held_out_85606_read"] = False
            handle.attrs["zperiod"] = 5
            handle.attrs["truth_layout"] = bool(truth_layout)
            handle.attrs["source_input_sha256"] = source_input_sha256
            handle.create_dataset("frame_index", data=frame_indices)
            handle.create_dataset(
                "phi",
                data=np.asarray(values, dtype=np.float64),
                chunks=(1, *NATIVE_SHAPE),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("truth-replay tolerances must be nonnegative")
    for path in (args.input, args.output_phi, args.output_result, *args.bout_output):
        assert_development_path(path)
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for compiled output") from error

    input_path = args.input.resolve(strict=True)
    bout_paths = [path.resolve(strict=True) for path in args.bout_output]
    with h5py.File(input_path, "r") as source:
        frame_key = "frame_index" if args.truth_layout else "coordinates/frame_index"
        frame_indices = np.asarray(source[frame_key][:], dtype=np.int64)
        if (
            frame_indices.ndim != 1
            or frame_indices.size == 0
            or np.any(np.diff(frame_indices) != 1)
            or frame_indices[0] < 0
            or frame_indices[-1] >= 624
        ):
            raise ValueError("input frame indices violate the 85604 contract")
        reference_phi = (
            np.asarray(source["phi"][:], dtype=np.float64)
            if args.truth_layout
            else None
        )
    if reference_phi is not None and reference_phi.shape != (
        frame_indices.size,
        *NATIVE_SHAPE,
    ):
        raise ValueError("truth-layout phi has the wrong shape")

    derived: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    with ExitStack() as stack:
        outputs = [
            stack.enter_context(netCDF4.Dataset(path, "r"))
            for path in bout_paths
        ]
        decomposition = []
        for path, output in zip(bout_paths, outputs):
            record = {
                "path": str(path),
                "NXPE": scalar_integer(output, "NXPE"),
                "NYPE": scalar_integer(output, "NYPE"),
                "MYSUB": scalar_integer(output, "MYSUB"),
                "PE_XIND": scalar_integer(output, "PE_XIND"),
                "PE_YIND": scalar_integer(output, "PE_YIND"),
            }
            if (
                record["NXPE"] != 1
                or record["NYPE"] != 4
                or record["MYSUB"] != 8
                or record["PE_XIND"] != 0
                or scalar_integer(output, "paper0_zperiod") != 5
                or scalar_integer(
                    output, "paper0_boundary_only_zero_interior_seed"
                )
                != 1
                or scalar_integer(output, "paper0_truth_layout")
                != int(args.truth_layout)
                or scalar_integer(output, "paper0_frame_count")
                != frame_indices.size
            ):
                raise ValueError(f"compiled output metadata differs: {record}")
            decomposition.append(record)
        if sorted(record["PE_YIND"] for record in decomposition) != [0, 1, 2, 3]:
            raise ValueError("compiled outputs do not cover four y partitions")

        for position, frame in enumerate(frame_indices):
            label = f"f{int(frame):03d}"
            for output in outputs:
                if scalar_integer(output, f"canonical_frame_index_{label}") != frame:
                    raise ValueError(f"compiled frame marker differs for {label}")
            values = assembled_xyz(
                outputs,
                decomposition,
                f"derived_phi_{label}",
            )[MODEL_X_SLICE]
            if values.shape != NATIVE_SHAPE or not np.all(np.isfinite(values)):
                raise ValueError(f"compiled derived phi is invalid for {label}")
            derived.append(values)
            if reference_phi is not None:
                per_frame.append(
                    {
                        "frame_index": int(frame),
                        **frame_metrics(
                            reference_phi[position],
                            values,
                            atol=args.atol,
                            rtol=args.rtol,
                        ),
                    }
                )

    source_hash = sha256_path(input_path)
    output_phi = args.output_phi.resolve(strict=False)
    _write_phi(
        output_phi,
        frame_indices=frame_indices,
        values=derived,
        source_input_sha256=source_hash,
        truth_layout=args.truth_layout,
    )
    truth_gate = None
    if reference_phi is not None:
        truth_gate = {
            "atol": args.atol,
            "rtol": args.rtol,
            "all_frames_passed": all(record["passes"] for record in per_frame),
            "maximum_absolute_difference": max(
                record["maximum_absolute_difference"] for record in per_frame
            ),
            "maximum_relative_l2": max(record["relative_l2"] for record in per_frame),
            "per_frame": per_frame,
        }
    result = {
        "schema_version": 1,
        "scope": "phase2_matched_e6b_elliptic_output",
        "status": "completed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "truth_layout": bool(args.truth_layout),
        "frame_interval": [int(frame_indices[0]), int(frame_indices[-1]) + 1],
        "frame_count": int(frame_indices.size),
        "source_input": {"path": str(input_path), "sha256": source_hash},
        "bout_outputs": [
            {"path": str(path), "sha256": sha256_path(path)} for path in bout_paths
        ],
        "derived_phi": {
            "path": str(output_phi),
            "sha256": sha256_path(output_phi),
            "dtype": "float64",
            "shape": [int(frame_indices.size), *NATIVE_SHAPE],
        },
        "truth_replay_gate": truth_gate,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_strict_json_atomic(args.output_result.resolve(strict=False), result)
    if truth_gate is not None and not truth_gate["all_frames_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
