#!/usr/bin/env python3
"""Evaluate one frozen 85604 frame shard for native/resampled sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.resampling import (  # noqa: E402
    paired_frame_metrics,
    paired_sufficient_statistics,
    periodic_resample_float32,
    relative_l2,
)
from tcv_diagnostics.transport import (  # noqa: E402
    SingleNullTopology,
    divergence_from_radial_face_flow_partial,
    radial_exb_face_flow_partial,
    toroidal_wedge_spacing,
)


C5P_FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
C5T_FIELDS = ("Ne", "Te", "Ti", "phi", "Vi")
RAW_ORACLE_FIELDS = ("Ne", "Pe", "Pi", "phi")
ADVECTED_FIELDS = ("Ne", "Pe", "Pi")
PRIMARY_QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)
COMPARISON_CATEGORIES = (
    "face_total",
    "face_xz",
    "face_xy",
    "divergence_total",
)
SELECTED_RAW_FRAMES = (0, 156, 312, 467, 623)
MODEL_X_SLICE = slice(2, 66)
TOPOLOGY = SingleNullTopology(
    separatrix_x_index=16,
    core_lower_y=8,
    core_upper_y=23,
    pfr_lower_y=7,
    pfr_upper_y=24,
)
SHARD_INTERVALS = tuple(
    [(start, min(start + 40, 500)) for start in range(0, 500, 40)]
    + [(start, min(start + 40, 624)) for start in range(500, 624, 40)]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def decode_strings(values: Any) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def assert_development_path(path: Path) -> None:
    if "85606" in path.parts or "85606" in str(path):
        raise ValueError(f"held-out path is prohibited: {path}")


def source_for_frame(
    sources: list[dict[str, Any]], frame_index: int
) -> tuple[dict[str, Any], int]:
    matches = [
        source
        for source in sources
        if int(source["global_start_inclusive"])
        <= frame_index
        < int(source["global_stop_exclusive"])
    ]
    if len(matches) != 1:
        raise ValueError(f"frame {frame_index} has {len(matches)} source matches")
    source = matches[0]
    return source, frame_index - int(source["global_start_inclusive"])


def verify_well_file(
    handle: Any,
    source: Mapping[str, Any],
    *,
    z_samples: int,
    required_fields: tuple[str, ...],
) -> None:
    first = int(source["global_start_inclusive"])
    stop = int(source["global_stop_exclusive"])
    expected_time = stop - first
    if str(handle.attrs.get("dataset_name")) not in {"TCV_85604", "TCV_c5"}:
        raise ValueError("unexpected Well dataset_name")
    if "t0_fields" not in handle or "dimensions/z" not in handle:
        raise ValueError("Well file lacks required groups")
    group = handle["t0_fields"]
    declared_fields = decode_strings(group.attrs["field_names"])
    for field in required_fields:
        if field not in declared_fields or field not in group:
            raise ValueError(f"Well file lacks required field {field}")
        dataset = group[field]
        expected_shape = (1, expected_time, 64, 32, z_samples)
        if tuple(dataset.shape) != expected_shape:
            raise ValueError(
                f"{field} shape {dataset.shape} differs from {expected_shape}"
            )
        if np.dtype(dataset.dtype) != np.dtype(np.float32):
            raise ValueError(f"{field} must use float32 storage")
    z_coordinate = np.asarray(handle["dimensions/z"][:])
    if z_coordinate.shape != (z_samples,):
        raise ValueError("Well z-coordinate length differs from expected")


def load_geometry(grid: Any) -> tuple[dict[str, np.ndarray], np.ndarray]:
    geometry = {
        name: np.asarray(
            np.ma.filled(grid.variables[source][:], np.nan), dtype=np.float64
        )[MODEL_X_SLICE]
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
    for name, values in geometry.items():
        if values.shape != (64, 32) or not np.all(np.isfinite(values)):
            raise ValueError(f"geometry {name} is invalid: {values.shape}")
    shift_angle = np.asarray(
        np.ma.filled(grid.variables["ShiftAngle"][:], np.nan), dtype=np.float64
    )[MODEL_X_SLICE]
    if shift_angle.shape != (64,) or not np.all(np.isfinite(shift_angle[:16])):
        raise ValueError("ShiftAngle is invalid on the used inner branch")
    return geometry, shift_angle


def primary_quantity_map(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    electron = 1.5 * np.asarray(arrays["Pe"], dtype=np.float64)
    ion = 1.5 * np.asarray(arrays["Pi"], dtype=np.float64)
    return {
        "particle": np.asarray(arrays["Ne"], dtype=np.float64),
        "electron_internal_energy": electron,
        "ion_internal_energy": ion,
        "total_internal_energy": electron + ion,
    }


def compute_transport_bundle(
    fields: Mapping[str, np.ndarray],
    geometry: Mapping[str, np.ndarray],
    shift_angle: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute all frozen numerical flow quantities on one toroidal grid."""

    phi = np.asarray(fields["phi"])
    if phi.shape[:2] != (64, 32):
        raise ValueError(f"potential has unexpected shape {phi.shape}")
    n_z = int(phi.shape[-1])
    face_total: dict[str, np.ndarray] = {}
    face_xz: dict[str, np.ndarray] = {}
    face_xy: dict[str, np.ndarray] = {}
    divergence_total: dict[str, np.ndarray] = {}
    for field in ADVECTED_FIELDS:
        q = np.asarray(fields[field])
        if q.shape != phi.shape:
            raise ValueError(f"{field} and phi shapes differ")
        faces = radial_exb_face_flow_partial(
            q,
            phi,
            geometry["jacobian"],
            geometry["g11"],
            geometry["g23"],
            geometry["bxy"],
            geometry["z_shift"],
            geometry["dy"],
            shift_angle,
            dz=toroidal_wedge_spacing(n_z, zperiod=5),
            topology=TOPOLOGY,
            zperiod=5,
            positive=True,
        )
        divergence = divergence_from_radial_face_flow_partial(
            faces, geometry["jacobian"], dx=geometry["dx"]
        )
        if not np.array_equal(faces.left_cell_indices, np.arange(1, 62)):
            raise ValueError("face index scope differs from protocol")
        if not np.array_equal(divergence.cell_indices, np.arange(2, 62)):
            raise ValueError("divergence cell scope differs from protocol")
        if not np.all(faces.valid_mask[:, 1:31]):
            raise ValueError("face validity mask excludes an interior point")
        if not np.all(divergence.valid_mask[:, 1:31]):
            raise ValueError("divergence validity mask excludes an interior point")
        face_total[field] = np.asarray(faces.flow[:, 1:31], dtype=np.float64)
        face_xz[field] = np.asarray(faces.xz_flow[:, 1:31], dtype=np.float64)
        face_xy[field] = np.asarray(faces.xy_flow[:, 1:31], dtype=np.float64)
        divergence_total[field] = np.asarray(
            divergence.divergence[:, 1:31], dtype=np.float64
        )
    bundle = {
        "face_total": primary_quantity_map(face_total),
        "face_xz": primary_quantity_map(face_xz),
        "face_xy": primary_quantity_map(face_xy),
        "divergence_total": primary_quantity_map(divergence_total),
    }
    for category, quantities in bundle.items():
        for quantity, values in quantities.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{category}/{quantity} contains non-finite values")
    return bundle


def compare_bundles(
    reference: Mapping[str, Mapping[str, np.ndarray]],
    candidate: Mapping[str, Mapping[str, np.ndarray]],
    *,
    resample_reference_to: int | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    comparison: dict[str, dict[str, dict[str, Any]]] = {}
    for category in COMPARISON_CATEGORIES:
        comparison[category] = {}
        for quantity in PRIMARY_QUANTITIES:
            reference_values = np.asarray(reference[category][quantity])
            if resample_reference_to is not None:
                reference_values = periodic_resample_float32(
                    reference_values, resample_reference_to
                )
            candidate_values = np.asarray(candidate[category][quantity])
            comparison[category][quantity] = paired_frame_metrics(
                reference_values, candidate_values, z_axis=-1
            )
    return comparison


def update_field_stream_digest(
    digest: Any, field: str, frame_index: int, values: np.ndarray
) -> None:
    header = json.dumps(
        {"field": field, "frame_index": frame_index},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values, dtype="<f4").tobytes(order="C"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.shard_count != len(SHARD_INTERVALS):
        raise ValueError(f"shard count must be {len(SHARD_INTERVALS)}")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index is outside the declared range")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
    ):
        raise ValueError("resampling audit requires the frozen 85604-only manifest")
    for source_group in ("native_81", "legacy_c5t_88"):
        for source in manifest["sources"][source_group]:
            assert_development_path(Path(source["path"]))
    raw_path = Path(manifest["sources"]["raw_float64_selected_frames"]["path"])
    grid_path = Path(manifest["sources"]["geometry"]["path"])
    assert_development_path(raw_path)
    assert_development_path(grid_path)

    try:
        import h5py
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency gate
        raise RuntimeError("h5py and netCDF4 are required") from error

    native_sources = manifest["sources"]["native_81"]
    legacy_sources = manifest["sources"]["legacy_c5t_88"]
    native_handles = {
        source["path"]: h5py.File(source["path"], "r")
        for source in native_sources
    }
    legacy_handles = {
        source["path"]: h5py.File(source["path"], "r")
        for source in legacy_sources
    }
    try:
        for source in native_sources:
            verify_well_file(
                native_handles[source["path"]],
                source,
                z_samples=81,
                required_fields=tuple(dict.fromkeys(C5P_FIELDS + C5T_FIELDS)),
            )
        for source in legacy_sources:
            verify_well_file(
                legacy_handles[source["path"]],
                source,
                z_samples=88,
                required_fields=C5T_FIELDS,
            )
        with netCDF4.Dataset(grid_path, "r") as grid:
            geometry, shift_angle = load_geometry(grid)
        with netCDF4.Dataset(raw_path, "r") as raw_file:
            raw_frame_indices = np.asarray(
                raw_file.variables["frame_index"][:], dtype=np.int64
            )
            if not np.array_equal(raw_frame_indices, SELECTED_RAW_FRAMES):
                raise ValueError("raw oracle frame indices differ from protocol")
            raw_position = {
                int(frame): position
                for position, frame in enumerate(raw_frame_indices.tolist())
            }

            first, stop = SHARD_INTERVALS[args.shard_index]
            field_digests = {field: hashlib.sha256() for field in C5P_FIELDS}
            frame_records: list[dict[str, Any]] = []
            for frame_index in range(first, stop):
                native_source, native_local = source_for_frame(
                    native_sources, frame_index
                )
                native_handle = native_handles[native_source["path"]]
                native = {
                    field: np.asarray(
                        native_handle["t0_fields"][field][0, native_local],
                        dtype=np.float32,
                    )
                    for field in C5P_FIELDS
                }
                for field, values in native.items():
                    if values.shape != (64, 32, 81) or not np.all(
                        np.isfinite(values)
                    ):
                        raise ValueError(f"invalid native {field} frame {frame_index}")
                    update_field_stream_digest(
                        field_digests[field], field, frame_index, values
                    )

                upsampled = {
                    field: periodic_resample_float32(values, 88)
                    for field, values in native.items()
                }
                round_trip = {
                    field: periodic_resample_float32(values, 81)
                    for field, values in upsampled.items()
                }
                field_round_trip = {
                    field: {
                        "relative_l2": relative_l2(native[field], round_trip[field]),
                        "sufficient_statistics": paired_sufficient_statistics(
                            native[field], round_trip[field]
                        ),
                    }
                    for field in C5P_FIELDS
                }

                native_bundle = compute_transport_bundle(
                    native, geometry, shift_angle
                )
                round_trip_bundle = compute_transport_bundle(
                    round_trip, geometry, shift_angle
                )
                direct_88_bundle = compute_transport_bundle(
                    upsampled, geometry, shift_angle
                )
                comparisons: dict[str, Any] = {
                    "round_trip": compare_bundles(
                        native_bundle,
                        round_trip_bundle,
                        resample_reference_to=None,
                    ),
                    "direct_88": compare_bundles(
                        native_bundle,
                        direct_88_bundle,
                        resample_reference_to=88,
                    ),
                    "raw64_vs_float32": None,
                }

                structural: dict[str, Any] = {
                    "selected_raw_equals_native_after_float32_cast": None,
                    "selected_legacy_c5t_resampling_bitwise_exact": None,
                }
                if frame_index in raw_position:
                    position = raw_position[frame_index]
                    raw = {
                        field: np.asarray(
                            raw_file.variables[field][position], dtype=np.float64
                        )
                        for field in RAW_ORACLE_FIELDS
                    }
                    raw_equal = {
                        field: bool(
                            np.array_equal(raw[field].astype(np.float32), native[field])
                        )
                        for field in RAW_ORACLE_FIELDS
                    }
                    if not all(raw_equal.values()):
                        raise ValueError(
                            f"raw/native float32 provenance failed at frame {frame_index}"
                        )
                    raw_bundle = compute_transport_bundle(raw, geometry, shift_angle)
                    comparisons["raw64_vs_float32"] = compare_bundles(
                        raw_bundle,
                        native_bundle,
                        resample_reference_to=None,
                    )
                    structural[
                        "selected_raw_equals_native_after_float32_cast"
                    ] = raw_equal

                    legacy_source, legacy_local = source_for_frame(
                        legacy_sources, frame_index
                    )
                    legacy_handle = legacy_handles[legacy_source["path"]]
                    legacy_exact: dict[str, bool] = {}
                    for field in C5T_FIELDS:
                        native_c5t = np.asarray(
                            native_handle["t0_fields"][field][0, native_local],
                            dtype=np.float32,
                        )
                        reproduced = periodic_resample_float32(native_c5t, 88)
                        stored = np.asarray(
                            legacy_handle["t0_fields"][field][0, legacy_local],
                            dtype=np.float32,
                        )
                        legacy_exact[field] = bool(np.array_equal(reproduced, stored))
                    if not all(legacy_exact.values()):
                        raise ValueError(
                            f"legacy z88 reproduction failed at frame {frame_index}"
                        )
                    structural[
                        "selected_legacy_c5t_resampling_bitwise_exact"
                    ] = legacy_exact

                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "native_source_path": native_source["path"],
                        "native_local_index": native_local,
                        "field_round_trip": field_round_trip,
                        "comparisons": comparisons,
                        "structural_checks": structural,
                    }
                )

        result = {
            "schema_version": 1,
            "phase": "phase2_85604_resampling_rank_shard",
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "development_run": "85604",
            "held_out_85606_read": False,
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "rank_shard_completed": True,
            "audit_completed": False,
            "shard": {
                "index": args.shard_index,
                "count": args.shard_count,
                "global_start_inclusive": first,
                "global_stop_exclusive": stop,
                "frame_count": stop - first,
            },
            "field_stream_sha256": {
                field: digest.hexdigest() for field, digest in field_digests.items()
            },
            "frame_records": frame_records,
        }
        strict_json_write(args.output, result)
        print(f"Wrote resampling shard {args.shard_index}/{args.shard_count}: {args.output}")
    finally:
        for handle in native_handles.values():
            handle.close()
        for handle in legacy_handles.values():
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
