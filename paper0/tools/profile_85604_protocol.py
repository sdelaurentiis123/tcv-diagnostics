#!/usr/bin/env python3
"""Execute the frozen Phase 1 profile on the single 85604 trajectory.

The tool reads full fields in bounded chunks, refuses sequestered paths and
existing outputs, and emits one compact standard-JSON result. It does not write
new datasets or inspect shot 85606.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.data_protocol import (  # noqa: E402
    C5_FIELDS,
    DEFAULT_SPLIT,
    FIELD_TRANSFORMS,
    RunningMoments,
    model_transform,
    operational_steady_screen,
    path_is_allowed,
    pattern_autocorrelation,
    representative_decorrelation,
    standardize,
    summarize_autocorrelation,
)
from tcv_diagnostics.well import VirtualWellTrajectory  # noqa: E402


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--chunk-frames", type=int, default=8)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    return value


def _singleton_attribute(value: Any, *, name: str) -> Any:
    """Unwrap scalar or length-one HDF5 attributes without hiding ambiguity."""

    current = value
    while isinstance(current, (list, tuple)):
        if len(current) != 1:
            raise ValueError(f"attribute {name} must be scalar, got {current!r}")
        current = current[0]
    return current


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {value} in {path}")
            ),
        )


def sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    )
    return {"commit": commit, "dirty": bool(status), "porcelain": status.splitlines()}


def _all_source_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = manifest["sources"]
    records: list[dict[str, Any]] = []
    for name, value in sources.items():
        values = value if isinstance(value, list) else [value]
        for index, record in enumerate(values):
            item = dict(record)
            item["source_name"] = name if len(values) == 1 else f"{name}[{index}]"
            records.append(item)
    return records


def verify_sources(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = []
    for record in _all_source_records(manifest):
        expected = record["sha256"]
        if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
            raise ValueError(f"invalid expected SHA-256 for {record['source_name']}")
        path = Path(record["path"]).expanduser().resolve(strict=True)
        if not path_is_allowed(path):
            raise ValueError(f"refusing sequestered source path: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path}: got {actual}, expected {expected}"
            )
        results.append(
            {
                "source_name": record["source_name"],
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return results


def _scalar(handle: h5py.File, name: str) -> float:
    value = np.asarray(handle[name][()])
    if value.size != 1:
        raise ValueError(f"expected scalar {name}, got shape {value.shape}")
    return float(value.reshape(-1)[0])


def inspect_raw(
    raw_path: Path,
    expected: Mapping[str, Any],
    global_time: np.ndarray,
) -> dict[str, Any]:
    requested_fields = list(expected["c5_fields"]) + list(
        expected["additional_state_candidates"]
    )
    with h5py.File(raw_path, "r") as handle:
        raw_time = np.asarray(handle["t_array"][...], dtype=np.float64).reshape(-1)
        zperiod = int(round(_scalar(handle, "zperiod")))
        zmin = _scalar(handle, "ZMIN")
        zmax = _scalar(handle, "ZMAX")
        omega_ci = _scalar(handle, "Omega_ci")
        metadata: dict[str, Any] = {}
        missing_candidates: list[str] = []
        for field in requested_fields:
            if field not in handle:
                missing_candidates.append(field)
                continue
            dataset = handle[field]
            metadata[field] = {
                "shape": [int(size) for size in dataset.shape],
                "dtype": str(dataset.dtype),
                "attrs": {
                    str(key): _jsonable(value) for key, value in dataset.attrs.items()
                },
            }

    if raw_time.shape != global_time.shape or not np.array_equal(raw_time, global_time):
        raise ValueError("raw and Well global time vectors do not match exactly")
    if raw_time.size != int(expected["total_frames"]):
        raise ValueError("raw frame count disagrees with manifest")
    if not math.isclose(float(raw_time[0]), float(expected["raw_time_first"])):
        raise ValueError("raw first time disagrees with manifest")
    if not math.isclose(float(raw_time[-1]), float(expected["raw_time_last"])):
        raise ValueError("raw last time disagrees with manifest")
    differences = np.diff(raw_time)
    if not np.allclose(
        differences,
        float(expected["normalized_frame_step"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("raw time cadence disagrees with manifest")
    if zperiod != int(expected["zperiod"]):
        raise ValueError(f"zperiod={zperiod}, expected {expected['zperiod']}")
    if not math.isclose(zmin, float(expected["zmin"]), abs_tol=1e-12):
        raise ValueError("ZMIN disagrees with manifest")
    if not math.isclose(zmax, float(expected["zmax"]), abs_tol=1e-12):
        raise ValueError("ZMAX disagrees with manifest")
    if not math.isclose(
        omega_ci, float(expected["omega_ci_per_second"]), rel_tol=1e-12
    ):
        raise ValueError("Omega_ci disagrees with manifest")

    field_checks: dict[str, Any] = {}
    for field, expectation in expected["raw_field_metadata"].items():
        if field not in metadata:
            raise ValueError(f"raw representative file lacks required C5 field {field}")
        attrs = metadata[field]["attrs"]
        actual_units = _singleton_attribute(
            attrs.get("units"), name=f"{field}.units"
        )
        actual_conversion = _singleton_attribute(
            attrs.get("conversion"), name=f"{field}.conversion"
        )
        units_ok = actual_units == expectation["units"]
        conversion_ok = actual_conversion is not None and math.isclose(
            float(actual_conversion), float(expectation["conversion"]), rel_tol=1e-12
        )
        if not units_ok or not conversion_ok:
            raise ValueError(
                f"raw metadata mismatch for {field}: units={actual_units!r}, "
                f"conversion={actual_conversion!r}"
            )
        field_checks[field] = {
            "units_match": units_ok,
            "conversion_match": conversion_ok,
        }

    cadence_microseconds = float(differences[0] / omega_ci * 1e6)
    if not math.isclose(
        cadence_microseconds,
        float(expected["cadence_microseconds"]),
        rel_tol=1e-12,
    ):
        raise ValueError("physical frame cadence disagrees with manifest")
    return {
        "time": {
            "frames": int(raw_time.size),
            "first_normalized": float(raw_time[0]),
            "last_normalized": float(raw_time[-1]),
            "normalized_step": float(differences[0]),
            "omega_ci_per_second": omega_ci,
            "cadence_microseconds": cadence_microseconds,
        },
        "toroidal_domain": {
            "zperiod": zperiod,
            "zmin": zmin,
            "zmax": zmax,
            "stored_torus_fraction": zmax - zmin,
            "mode_mapping": f"n = {zperiod}k",
        },
        "field_metadata": metadata,
        "required_field_checks": field_checks,
        "missing_additional_candidates": [
            field
            for field in missing_candidates
            if field in expected["additional_state_candidates"]
        ],
    }


def inspect_bout_settings(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def option(name: str) -> str:
        match = re.search(rf"(?mi)^\s*{re.escape(name)}\s*=\s*([^#\n]+)", text)
        if match is None:
            raise ValueError(f"could not find {name} in {path}")
        return match.group(1).strip()

    version = option("version")
    revision = option("revision")
    if version != expected["bout_version"] or revision != expected["bout_revision"]:
        raise ValueError(
            f"BOUT identity mismatch: version={version}, revision={revision}"
        )
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for line in text.splitlines():
        heading = re.fullmatch(r"\s*\[([^]]+)\]\s*", line)
        if heading is not None:
            current_section = heading.group(1)
            sections.setdefault(current_section, [])
        elif current_section is not None:
            sections[current_section].append(line)

    def section(name: str) -> str:
        if name not in sections:
            raise ValueError(f"could not find [{name}] section in {path}")
        return "\n".join(sections[name])

    electron = section("e")
    vorticity = section("vorticity")
    return {
        "version": version,
        "revision": revision,
        "electron_momentum_evolved": bool(
            re.search(r"(?mi)^\s*type\s*=.*\bevolve_momentum\b", electron)
        ),
        "vorticity_component_present": True,
        "diamagnetic_polarisation_enabled": bool(
            re.search(
                r"(?mi)^\s*diamagnetic_polarisation\s*=\s*true", vorticity
            )
        ),
        "exb_advection_simplified_false": bool(
            re.search(
                r"(?mi)^\s*exb_advection_simplified\s*=\s*false", vorticity
            )
        ),
    }


def profile_fields(
    trajectory: VirtualWellTrajectory,
    *,
    chunk_frames: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, float | int]],
]:
    frame_means: dict[str, np.ndarray] = {}
    fluctuation_rms: dict[str, np.ndarray] = {}
    normalization: dict[str, dict[str, float | int]] = {}

    for field in C5_FIELDS:
        means: list[np.ndarray] = []
        rms_values: list[np.ndarray] = []
        moments = RunningMoments()
        global_cursor = 0
        for chunk in trajectory.iter_field_chunks(
            field, 0, trajectory.total_frames, chunk_frames=chunk_frames
        ):
            transformed = model_transform(field, np.asarray(chunk, dtype=np.float64))
            flat = transformed.reshape(transformed.shape[0], -1)
            chunk_means = np.mean(flat, axis=1, dtype=np.float64)
            centered = flat - chunk_means[:, None]
            chunk_rms = np.sqrt(np.mean(centered * centered, axis=1, dtype=np.float64))
            means.append(chunk_means)
            rms_values.append(chunk_rms)

            training_stop = min(
                transformed.shape[0], DEFAULT_SPLIT.train.stop - global_cursor
            )
            if training_stop > 0:
                moments.update(transformed[:training_stop])
            global_cursor += transformed.shape[0]

        if global_cursor != trajectory.total_frames:
            raise RuntimeError(f"read {global_cursor} frames for {field}")
        frame_means[field] = np.concatenate(means)
        fluctuation_rms[field] = np.concatenate(rms_values)
        stats = moments.finalize()
        expected_count = DEFAULT_SPLIT.train.frames * int(
            np.prod(trajectory.spatial_shape, dtype=np.int64)
        )
        if stats["count"] != expected_count:
            raise RuntimeError(
                f"normalization count {stats['count']} != expected {expected_count}"
            )
        normalization[field] = {
            **stats,
            "transform": FIELD_TRANSFORMS[field],
            "fit_start_inclusive": DEFAULT_SPLIT.train.start,
            "fit_stop_exclusive": DEFAULT_SPLIT.train.stop,
        }
    return frame_means, fluctuation_rms, normalization


def compute_decorrelation(
    trajectory: VirtualWellTrajectory,
    normalization: Mapping[str, Mapping[str, float | int]],
    cadence_microseconds: float,
    *,
    chunk_frames: int,
) -> dict[str, Any]:
    per_field: dict[str, Any] = {}
    for field in C5_FIELDS:
        frames = trajectory.read_field(
            field,
            DEFAULT_SPLIT.train.start,
            DEFAULT_SPLIT.train.stop,
            chunk_frames=chunk_frames,
            strides=(4, 2, 4),
        )
        transformed = model_transform(field, np.asarray(frames, dtype=np.float64))
        normalized = standardize(
            transformed,
            float(normalization[field]["mean"]),
            float(normalization[field]["std"]),
        )
        curve = pattern_autocorrelation(normalized, max_lag=108)
        per_field[field] = summarize_autocorrelation(curve, cadence_microseconds)
    return {
        "definition": "uniform-grid Eulerian fluctuation-pattern autocorrelation",
        "training_indices": [
            DEFAULT_SPLIT.train.start,
            DEFAULT_SPLIT.train.stop,
        ],
        "spatial_strides_xyz": [4, 2, 4],
        "max_lag_frames": 108,
        "per_field": per_field,
        "representative": representative_decorrelation(
            per_field, cadence_microseconds
        ),
    }


def main() -> None:
    args = parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("chunk-frames must be positive")
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve(strict=False)
    if not path_is_allowed(manifest_path) or not path_is_allowed(output):
        raise ValueError("refusing sequestered manifest or output path")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise ValueError(f"output parent is not a directory: {output.parent}")

    manifest = _read_json(manifest_path)
    if manifest["run_id"] != "85604" or manifest.get("blind_test_accessed") is not False:
        raise ValueError("Phase 1 manifest must be locked to 85604 with blind test closed")
    state = git_state()
    if state["commit"] != args.expected_commit:
        raise ValueError(
            f"Git commit {state['commit']} != expected {args.expected_commit}"
        )
    if state["dirty"]:
        raise ValueError(f"refusing dirty worktree: {state['porcelain']}")

    source_records = verify_sources(manifest)
    source_by_name = {record["source_name"]: record for record in source_records}
    well_records = manifest["sources"]["well_shards"]
    trajectory = VirtualWellTrajectory(
        [record["path"] for record in well_records], required_fields=C5_FIELDS
    )
    expected = manifest["expected"]
    if trajectory.total_frames != int(expected["total_frames"]):
        raise ValueError("Well trajectory frame count disagrees with manifest")
    if tuple(trajectory.spatial_shape) != tuple(expected["converted_spatial_shape"]):
        raise ValueError("Well spatial shape disagrees with manifest")
    for shard, record in zip(trajectory.shards, well_records, strict=True):
        if (
            shard.global_start != int(record["global_start_inclusive"])
            or shard.global_stop != int(record["global_stop_exclusive"])
        ):
            raise ValueError("Well shard global index mapping disagrees with manifest")

    raw_path = Path(manifest["sources"]["raw_representative"]["path"])
    raw = inspect_raw(raw_path, expected, trajectory.time)
    settings_path = Path(manifest["sources"]["bout_settings"]["path"])
    bout = inspect_bout_settings(settings_path, expected)

    frame_means, fluctuation_rms, normalization = profile_fields(
        trajectory, chunk_frames=args.chunk_frames
    )
    steady = operational_steady_screen(frame_means, fluctuation_rms)
    split_status = "frozen" if steady["passes"] else "blocked_by_steady_state_screen"
    decorrelation = None
    if steady["passes"]:
        decorrelation = compute_decorrelation(
            trajectory,
            normalization,
            float(raw["time"]["cadence_microseconds"]),
            chunk_frames=args.chunk_frames,
        )

    result = {
        "schema_version": "0.1.0",
        "phase": "phase1_immutable_data_protocol",
        "run_id": "85604",
        "blind_test_accessed": False,
        "protocol": {
            "path": str(ROOT / "paper0/protocol/PHASE1_DATA_PROTOCOL.md"),
            "frozen_rule_commit": manifest["protocol_commit"],
            "execution_commit": state["commit"],
        },
        "execution": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "h5py_version": h5py.__version__,
            "git": state,
            "chunk_frames": args.chunk_frames,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "verified_sources": source_records,
        "source_record_index": source_by_name,
        "trajectory": {
            "axes": {
                "well_field": ["trajectory", "time", "x", "y", "z"],
                "virtual_field": ["time", "x", "y", "z"],
            },
            "total_frames": trajectory.total_frames,
            "spatial_shape_xyz": list(trajectory.spatial_shape),
            "well_time_first": float(trajectory.time[0]),
            "well_time_last": float(trajectory.time[-1]),
            "well_time_step": float(np.diff(trajectory.time)[0]),
            "shards": [
                {
                    "path": str(shard.path),
                    "global_start_inclusive": shard.global_start,
                    "global_stop_exclusive": shard.global_stop,
                    "frames": shard.frames,
                }
                for shard in trajectory.shards
            ],
        },
        "raw_simulation": raw,
        "bout_build_and_equation_flags": bout,
        "field_protocol": {
            "c5_legacy_observable_baseline": list(C5_FIELDS),
            "c5_assumed_markov_complete": False,
            "additional_state_candidates": expected["additional_state_candidates"],
            "model_transforms": FIELD_TRANSFORMS,
            "physical_time_policy": manifest["time_conditioning"],
        },
        "steady_state_screen": steady,
        "temporal_profiles": {
            "definition": "model-coordinate uniform-grid summaries used by the frozen screen",
            "frame_indices": list(range(trajectory.total_frames)),
            "normalized_simulation_time": [
                float(value) for value in trajectory.time
            ],
            "spatial_mean_by_field": {
                field: [float(value) for value in frame_means[field]]
                for field in C5_FIELDS
            },
            "fluctuation_rms_by_field": {
                field: [float(value) for value in fluctuation_rms[field]]
                for field in C5_FIELDS
            },
            "phi_spatial_mean_used_for_stationarity": False,
        },
        "split": {
            **DEFAULT_SPLIT.to_dict(),
            "status": split_status,
            "condition": manifest["split"]["freeze_condition"],
        },
        "training_only_normalization": normalization,
        "decorrelation": decorrelation,
        "phase1_learning_gate_open": bool(steady["passes"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(output),
        "split_status": split_status,
        "steady_state_passes": steady["passes"],
        "steady_state_failures": steady["failures"],
        "decorrelation_representative": (
            None if decorrelation is None else decorrelation["representative"]
        ),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
