#!/usr/bin/env python3
"""Build and verify one frozen 85604 model-dataset shard."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.model_data import (  # noqa: E402
    StreamingMoments,
    VOLUME_FIELDS,
    apply_moment_transform,
    array_sha256,
    assert_development_path,
    canonical_float64_sha256,
    finite_real_array,
    load_strict_json,
    relative_l2_by_frame,
    sha256_file,
    source_segments,
    validate_intervals,
    write_strict_json_atomic,
)
from tcv_diagnostics.resampling import periodic_resample_float32  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


def parse_sha256_record(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise ValueError(f"invalid SHA-256 preflight line in {path}: {line!r}")
        digest, raw_name = pieces
        name = raw_name.lstrip("*")
        if name in records:
            raise ValueError(f"duplicate preflight path {name}")
        records[name] = digest
    return records


def required_preflight_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for group in ("native_81_well", "legacy_z88_transform_oracle"):
        for source in manifest["sources"][group]:
            expected[str(source["path"])] = str(source["sha256"])
    boundary = manifest["sources"]["potential_boundary"]
    expected[str(boundary["extraction_record_path"])] = str(
        boundary["extraction_record_sha256"]
    )
    return expected


def verify_source_preflight(
    path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    records = parse_sha256_record(path)
    expected = required_preflight_hashes(manifest)
    if records != expected:
        raise ValueError("source preflight record differs from the frozen source locks")
    return records


def verify_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_path: Path,
    shard_index: int,
) -> tuple[tuple[int, int], ...]:
    if (
        manifest["development_run"] != "85604"
        or manifest["sequestered_run"] != "85606"
        or manifest["held_out_85606_access_allowed"]
        or manifest["training_authorized"]
    ):
        raise ValueError("model conversion requires the frozen 85604-only manifest")
    if tuple(manifest["data"]["volume_fields"]) != VOLUME_FIELDS:
        raise ValueError("volume-field union differs from the frozen order")
    if manifest["data"]["zperiod"] != 5:
        raise ValueError("zperiod differs from the frozen value 5")
    if sha256_file(protocol_path) != manifest["protocol"]["sha256"]:
        raise ValueError("protocol hash differs from the manifest lock")
    if protocol_path.resolve() != (ROOT / manifest["protocol"]["path"]).resolve():
        raise ValueError("protocol path differs from the manifest lock")
    intervals = validate_intervals(
        manifest["output"]["shard_intervals"],
        expected_start=0,
        expected_stop=624,
    )
    if not 0 <= shard_index < len(intervals):
        raise ValueError("shard index is outside the frozen interval set")
    for group in ("native_81_well", "legacy_z88_transform_oracle"):
        for source in manifest["sources"][group]:
            assert_development_path(Path(source["path"]))
    assert_development_path(
        Path(manifest["sources"]["potential_boundary"]["extraction_record_path"])
    )
    return intervals


def read_well_field(
    sources: Sequence[Mapping[str, Any]],
    *,
    field: str,
    start: int,
    stop: int,
    expected_z: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for source, local_start, local_stop, _, _ in source_segments(
        sources, start, stop
    ):
        path = Path(source["path"])
        assert_development_path(path)
        with h5py.File(path, "r") as handle:
            dataset_name = source.get(
                "field_dataset_template", "t0_fields/{field}"
            ).format(field=field)
            if dataset_name not in handle:
                raise ValueError(f"{path} lacks {dataset_name}")
            dataset = handle[dataset_name]
            expected_shape = source.get("field_shape")
            if expected_shape is not None and list(dataset.shape) != expected_shape:
                raise ValueError(
                    f"{path}:{dataset_name} shape {dataset.shape} differs from "
                    f"{expected_shape}"
                )
            if dataset.dtype != np.dtype(np.float32):
                raise ValueError(f"{path}:{dataset_name} is not float32")
            pieces.append(
                np.asarray(dataset[0, local_start:local_stop], dtype=np.float32)
            )
    result = np.concatenate(pieces, axis=0)
    expected = (stop - start, 64, 32, expected_z)
    if result.shape != expected:
        raise ValueError(f"source {field} shape {result.shape} differs from {expected}")
    return np.asarray(finite_real_array(field, result), dtype=np.float32)


def read_well_time(
    sources: Sequence[Mapping[str, Any]],
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for source, local_start, local_stop, _, _ in source_segments(
        sources, start, stop
    ):
        path = Path(source["path"])
        with h5py.File(path, "r") as handle:
            name = source.get("time_dataset", "dimensions/time")
            if name not in handle:
                raise ValueError(f"{path} lacks {name}")
            values = np.asarray(handle[name][local_start:local_stop], dtype=np.float64)
            pieces.append(values)
    result = np.concatenate(pieces)
    if result.shape != (stop - start,) or not np.all(np.isfinite(result)):
        raise ValueError("source time has invalid shape or values")
    return result


def load_boundary(
    manifest: Mapping[str, Any],
    *,
    shard_index: int,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency gate
        raise RuntimeError("netCDF4 is required for the boundary source") from error

    specification = manifest["sources"]["potential_boundary"]
    record_path = Path(specification["extraction_record_path"])
    if sha256_file(record_path) != specification["extraction_record_sha256"]:
        raise ValueError("potential-boundary extraction record hash differs")
    record = load_strict_json(record_path)
    shards = record["shards"]
    if len(shards) != 8:
        raise ValueError("boundary extraction record does not contain eight shards")
    source = shards[shard_index]
    if (
        int(source["shard_index"]) != shard_index
        or int(source["start"]) != start
        or int(source["stop"]) != stop
    ):
        raise ValueError("boundary shard interval differs from model shard")
    canonical_path = Path(source[specification["canonical_file_key"]])
    assert_development_path(canonical_path)
    if sha256_file(canonical_path) != source[
        specification["canonical_file_sha256_key"]
    ]:
        raise ValueError("canonical boundary file hash differs")
    with netCDF4.Dataset(canonical_path, "r") as dataset:
        frame_index = np.asarray(dataset.variables["frame_index"][:], dtype=np.int64)
        time = np.asarray(dataset.variables["normalized_time"][:], dtype=np.float64)
        boundary64 = np.asarray(
            dataset.variables[specification["array_name"]][:], dtype=np.float64
        )
    expected_frames = np.arange(start, stop, dtype=np.int64)
    if not np.array_equal(frame_index, expected_frames):
        raise ValueError("canonical boundary frame indices differ")
    if boundary64.shape != (stop - start, 2, 32):
        raise ValueError(f"canonical boundary shape differs: {boundary64.shape}")
    finite_real_array("canonical Bphi", boundary64)
    expected_array_hash = source["array_sha256"][
        specification["array_sha256_key"]
    ]
    actual_array_hash = canonical_float64_sha256(boundary64)
    if actual_array_hash != expected_array_hash:
        raise ValueError("canonical boundary array hash differs")
    return boundary64, frame_index, time, {
        "extraction_record_sha256": specification["extraction_record_sha256"],
        "canonical_file": str(canonical_path),
        "canonical_file_sha256": source[
            specification["canonical_file_sha256_key"]
        ],
        "source_array_sha256": actual_array_hash,
    }


def create_output_file(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    protocol_sha256: str,
    paper0_commit: str,
    slurm_job_id: int,
    shard_index: int,
    start: int,
    stop: int,
    frame_index: np.ndarray,
    time: np.ndarray,
    volume_values: Mapping[str, np.ndarray],
    boundary: np.ndarray,
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite temporary output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as handle:
        handle.attrs["paper0_phase"] = manifest["phase"]
        handle.attrs["development_run"] = "85604"
        handle.attrs["held_out_85606_read"] = False
        handle.attrs["paper0_commit"] = paper0_commit
        handle.attrs["manifest_sha256"] = manifest_sha256
        handle.attrs["protocol_sha256"] = protocol_sha256
        handle.attrs["slurm_job_id"] = int(slurm_job_id)
        handle.attrs["shard_index"] = int(shard_index)
        handle.attrs["global_start_inclusive"] = int(start)
        handle.attrs["global_stop_exclusive"] = int(stop)
        handle.attrs["zperiod"] = 5
        handle.attrs["time_coordinate"] = "normalized_ion_cyclotron_time"
        fields = handle.create_group("fields")
        chunk_shape = tuple(manifest["output"]["volume_chunk_shape"])
        for field in VOLUME_FIELDS:
            values = np.asarray(volume_values[field], dtype=np.float32)
            fields.create_dataset(
                field,
                data=values,
                dtype="f4",
                chunks=chunk_shape,
                compression=None,
                shuffle=False,
                fletcher32=False,
            )
        boundary_group = handle.create_group("boundary")
        boundary_group.create_dataset(
            "Bphi",
            data=np.asarray(boundary, dtype=np.float32),
            dtype="f4",
            chunks=(1, 2, 32),
            compression=None,
        )
        coordinates = handle.create_group("coordinates")
        coordinates.create_dataset("frame_index", data=frame_index, dtype="i8")
        coordinates.create_dataset("time", data=time, dtype="f8")
        handle.flush()


def verify_reopened_output(
    path: Path,
    *,
    expected_digests: Mapping[str, str],
    expected_boundary_digest: str,
    frame_index: np.ndarray,
    time: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"fields": {}, "all_bitwise_exact": True}
    with h5py.File(path, "r") as handle:
        if int(handle.attrs["global_start_inclusive"]) != start:
            raise ValueError("reopened shard start differs")
        if int(handle.attrs["global_stop_exclusive"]) != stop:
            raise ValueError("reopened shard stop differs")
        for field in VOLUME_FIELDS:
            values = np.asarray(handle[f"fields/{field}"][...])
            digest = array_sha256(values)
            exact = digest == expected_digests[field]
            result["fields"][field] = {
                "array_sha256": digest,
                "writer_echo_bitwise_exact": exact,
            }
            result["all_bitwise_exact"] &= exact
        boundary = np.asarray(handle["boundary/Bphi"][...])
        boundary_digest = array_sha256(boundary)
        boundary_exact = boundary_digest == expected_boundary_digest
        result["boundary"] = {
            "array_sha256": boundary_digest,
            "writer_echo_bitwise_exact": boundary_exact,
        }
        result["all_bitwise_exact"] &= boundary_exact
        frame_exact = np.array_equal(
            np.asarray(handle["coordinates/frame_index"][:], dtype=np.int64),
            frame_index,
        )
        time_exact = np.array_equal(
            np.asarray(handle["coordinates/time"][:], dtype=np.float64),
            time,
        )
        result["coordinates"] = {
            "frame_index_bitwise_exact": frame_exact,
            "time_bitwise_exact": time_exact,
        }
        result["all_bitwise_exact"] &= frame_exact and time_exact
    if not result["all_bitwise_exact"]:
        raise ValueError("writer/reopen bitwise echo failed")
    return result


def main() -> int:
    args = parse_args()
    manifest = load_strict_json(args.manifest)
    intervals = verify_manifest(
        manifest,
        protocol_path=args.protocol,
        shard_index=args.shard_index,
    )
    preflight = verify_source_preflight(args.source_preflight, manifest)
    manifest_sha256 = sha256_file(args.manifest)
    protocol_sha256 = sha256_file(args.protocol)
    start, stop = intervals[args.shard_index]
    expected_frames = np.arange(start, stop, dtype=np.int64)

    output_specification = manifest["output"]
    expected_output_root = (
        Path(output_specification["base_directory"])
        / output_specification["job_directory_template"].format(
            slurm_job_id=args.slurm_job_id
        )
    )
    if args.output_root.resolve() != expected_output_root.resolve():
        raise ValueError(
            f"output root {args.output_root} differs from {expected_output_root}"
        )
    shard_directory = args.output_root / output_specification["shard_directory"]
    partial_directory = args.output_root / "partials"
    shard_name = output_specification["shard_filename_template"].format(
        shard_index=args.shard_index
    )
    partial_name = output_specification["partial_record_template"].format(
        shard_index=args.shard_index
    )
    final_path = shard_directory / shard_name
    partial_path = partial_directory / partial_name
    if final_path.exists() or partial_path.exists():
        raise FileExistsError("refusing to overwrite an existing shard artifact")
    temporary_path = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
    if temporary_path.exists():
        raise FileExistsError(f"refusing to overwrite {temporary_path}")

    native_sources = manifest["sources"]["native_81_well"]
    legacy_sources = manifest["sources"]["legacy_z88_transform_oracle"]
    time = read_well_time(native_sources, start=start, stop=stop)
    boundary64, boundary_frames, boundary_time, boundary_source = load_boundary(
        manifest,
        shard_index=args.shard_index,
        start=start,
        stop=stop,
    )
    if not np.array_equal(boundary_frames, expected_frames):
        raise ValueError("boundary frames differ from expected frames")
    if not np.array_equal(boundary_time, time):
        raise ValueError("Well and canonical boundary time coordinates differ")
    time_specification = manifest["data"]["time"]
    expected_time = time_specification["first"] + (
        time_specification["normalized_step"] * expected_frames
    )
    if not np.array_equal(time, expected_time):
        raise ValueError("shard normalized time differs from the frozen sequence")

    training_stop = int(manifest["paper0_split"]["normalization_fit_frames"][1])
    training_count = max(0, min(stop, training_stop) - start)
    transforms = manifest["normalization"]["transforms"]
    field_values: dict[str, np.ndarray] = {}
    field_records: dict[str, Any] = {}
    field_digests: dict[str, str] = {}
    normalization: dict[str, Any] = {}
    round_trip_limit = float(
        manifest["integrity_gates"]["field_round_trip_max_per_frame_relative_l2"]
    )
    legacy_fields = set(
        manifest["integrity_gates"]["legacy_z88_bitwise_fields"]
    )

    for field in VOLUME_FIELDS:
        native = read_well_field(
            native_sources,
            field=field,
            start=start,
            stop=stop,
            expected_z=81,
        )
        model = periodic_resample_float32(native, 88)
        if model.shape != (stop - start, 64, 32, 88):
            raise ValueError(f"resampled {field} has unexpected shape {model.shape}")
        round_trip = periodic_resample_float32(model, 81)
        errors = relative_l2_by_frame(native, round_trip)
        maximum_error = float(np.max(errors))
        if not math.isfinite(maximum_error) or maximum_error > round_trip_limit:
            raise ValueError(
                f"{field} round-trip maximum {maximum_error} exceeds "
                f"{round_trip_limit}"
            )
        legacy_exact: bool | None = None
        legacy_mismatch_count: int | None = None
        if field in legacy_fields:
            legacy = read_well_field(
                legacy_sources,
                field=field,
                start=start,
                stop=stop,
                expected_z=88,
            )
            if legacy.shape != model.shape:
                raise ValueError(f"legacy {field} shape differs from model output")
            legacy_exact = bool(np.array_equal(model, legacy))
            legacy_mismatch_count = int(np.count_nonzero(model != legacy))
            if not legacy_exact:
                raise ValueError(
                    f"resampled {field} differs from the legacy z88 oracle"
                )
        digest = array_sha256(model)
        field_digests[field] = digest
        field_values[field] = model
        field_records[field] = {
            "source_dtype": str(native.dtype),
            "source_shape": list(native.shape),
            "output_dtype": str(model.dtype),
            "output_shape": list(model.shape),
            "output_array_sha256": digest,
            "round_trip": {
                "maximum_per_frame_relative_l2": maximum_error,
                "mean_per_frame_relative_l2": float(np.mean(errors)),
                "limit": round_trip_limit,
                "passed": True,
            },
            "legacy_z88": {
                "required": field in legacy_fields,
                "bitwise_exact": legacy_exact,
                "mismatch_count": legacy_mismatch_count,
            },
        }
        if training_count > 0:
            moments = StreamingMoments()
            moments.update(
                apply_moment_transform(
                    field,
                    model[:training_count],
                    transforms,
                )
            )
            normalization[field] = moments.finalize()

    boundary32 = np.asarray(boundary64, dtype=np.float32)
    boundary_cast_exact = np.array_equal(
        boundary32,
        boundary64.astype(np.float32),
    )
    if not boundary_cast_exact:
        raise ValueError("Bphi explicit float32 cast check failed")
    boundary_digest = array_sha256(boundary32)
    if training_count > 0:
        for side_index, side_name in enumerate(("inner", "outer")):
            moments = StreamingMoments()
            moments.update(
                apply_moment_transform(
                    "Bphi",
                    boundary32[:training_count, side_index, :],
                    transforms,
                )
            )
            normalization[f"Bphi/{side_name}"] = moments.finalize()

    try:
        create_output_file(
            temporary_path,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            protocol_sha256=protocol_sha256,
            paper0_commit=args.paper0_commit,
            slurm_job_id=args.slurm_job_id,
            shard_index=args.shard_index,
            start=start,
            stop=stop,
            frame_index=expected_frames,
            time=time,
            volume_values=field_values,
            boundary=boundary32,
        )
        echo = verify_reopened_output(
            temporary_path,
            expected_digests=field_digests,
            expected_boundary_digest=boundary_digest,
            frame_index=expected_frames,
            time=time,
            start=start,
            stop=stop,
        )
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite {final_path}")
        os.replace(temporary_path, final_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    result = {
        "schema_version": 1,
        "phase": "phase2_85604_model_dataset_shard",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_sha256,
        "source_preflight": {
            "path": str(args.source_preflight),
            "sha256": sha256_file(args.source_preflight),
            "records": preflight,
            "passed": True,
        },
        "shard": {
            "index": args.shard_index,
            "global_start_inclusive": start,
            "global_stop_exclusive": stop,
            "frame_count": stop - start,
            "frame_indices": expected_frames.tolist(),
            "normalized_time_first": float(time[0]),
            "normalized_time_last": float(time[-1]),
        },
        "boundary_source": boundary_source,
        "fields": field_records,
        "boundary": {
            "source_dtype": str(boundary64.dtype),
            "output_dtype": str(boundary32.dtype),
            "output_shape": list(boundary32.shape),
            "output_array_sha256": boundary_digest,
            "explicit_float32_cast_bitwise_exact": boundary_cast_exact,
        },
        "normalization_partial": {
            "training_local_frame_count": training_count,
            "records": normalization,
        },
        "writer_echo": echo,
        "output": {
            "path": str(final_path),
            "bytes": final_path.stat().st_size,
            "sha256": sha256_file(final_path),
        },
        "all_shard_gates_passed": True,
    }
    write_strict_json_atomic(partial_path, result)
    print(
        json.dumps(
            {
                "shard_index": args.shard_index,
                "frames": [start, stop],
                "output": str(final_path),
                "partial": str(partial_path),
                "all_shard_gates_passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
