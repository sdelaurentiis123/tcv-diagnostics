#!/usr/bin/env python3
"""Run the frozen 85604-only O1 geometry-aware codec-transport oracle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_transport import (  # noqa: E402
    STATE_PATHS,
    TRANSPORT_QUANTITIES,
    TransportComparisonAccumulator,
    build_codec_transport_geometry,
    build_o1_transport_gate,
    c5t_transport_state,
    direct_pressure_transport_state,
    evaluate_transport_state,
    per_frame_relative_l2,
)
from tcv_diagnostics.data_protocol import C5_FIELDS  # noqa: E402
from tcv_diagnostics.resampling import periodic_resample_float32  # noqa: E402
from tcv_diagnostics.transport import (  # noqa: E402
    SingleNullTopology,
    hermes_transport_scales,
    toroidal_wedge_spacing,
)
from tcv_diagnostics.well import VirtualWellTrajectory  # noqa: E402

from lola.autoencoder import get_autoencoder  # noqa: E402
from lola.data import field_postprocess, field_preprocess  # noqa: E402


LEGACY_MEAN = np.asarray([-1.9359, 0.9337, 1.2636, 2.8614, -0.1795])
LEGACY_STD = np.asarray([1.4488, 0.5312, 0.4681, 1.2784, 0.9219])
NATIVE_FIELDS = ("Ne", "Te", "Ti", "Pe", "Pi", "phi", "Vi")
TOTAL_FRAMES = 624
BLOCK_FRAMES = 78
NATIVE_SHAPE = (64, 32, 81)
CODEC_SHAPE = (64, 32, 88)
ZPERIOD = 5
CODEC_SPECS = {
    "f8": {
        "run_name": "w24x2ybf_tcv_c5_dcae_3d_tcv_f8c64",
        "latent_shape": (64, 8, 4, 11),
        "lineage": "from_scratch_mae_50x2048",
    },
    "z44": {
        "run_name": "z44c6604191_tcv_c5_dcae_3d_tcv_f8z2c64",
        "latent_shape": (64, 8, 4, 44),
        "lineage": "non_strict_z22_continuation_mae_plus_increment_12x1024",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--native-train-h5", required=True)
    parser.add_argument("--native-valid-h5", required=True)
    parser.add_argument("--legacy-train-h5", required=True)
    parser.add_argument("--legacy-valid-h5", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--f8-run", required=True)
    parser.add_argument("--z44-run", required=True)
    parser.add_argument("--prior-o1", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_file(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected file: {path}")
    if "85606" in str(path).lower() or any(
        part.lower() == "test" for part in path.parts
    ):
        raise ValueError(f"refusing sequestered input path: {path}")
    return path


def _resolved_run(path_text: str, codec_name: str) -> Path:
    path = Path(path_text).expanduser().resolve(strict=True)
    if not path.is_dir() or path.name != CODEC_SPECS[codec_name]["run_name"]:
        raise ValueError(f"unexpected {codec_name} codec directory: {path}")
    for filename in ("config.yaml", "state.pth"):
        if not (path / filename).is_file():
            raise ValueError(f"missing {path / filename}")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                _json_safe(record),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _configure_determinism() -> dict[str, Any]:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "seed": 0,
        "decode_noise": False,
        "dtype": "float32_codec_float64_transport",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _verify_sources(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    sources = manifest["sources"]
    paths = {
        "native_train": _resolved_file(args.native_train_h5),
        "native_valid": _resolved_file(args.native_valid_h5),
        "legacy_train": _resolved_file(args.legacy_train_h5),
        "legacy_valid": _resolved_file(args.legacy_valid_h5),
        "geometry": _resolved_file(args.geometry),
        "prior_o1": _resolved_file(args.prior_o1),
    }
    expected = {
        "native_train": sources["native_81"][0],
        "native_valid": sources["native_81"][1],
        "legacy_train": sources["legacy_c5t_88"][0],
        "legacy_valid": sources["legacy_c5t_88"][1],
        "geometry": sources["geometry"],
        "prior_o1": sources["prior_o1_compact"],
    }
    records: dict[str, Any] = {}
    for name, path in paths.items():
        actual_hash = sha256(path)
        if str(path) != str(Path(expected[name]["path"]).resolve(strict=False)):
            if name != "prior_o1":
                raise ValueError(f"{name} path differs from frozen manifest")
        if actual_hash != expected[name]["sha256"]:
            raise ValueError(f"{name} SHA-256 differs from frozen manifest")
        records[name] = {"path": str(path), "sha256": actual_hash}
    return {"paths": paths, "records": records}


def _verify_codec_run(
    codec_name: str,
    run: Path,
    manifest: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    frozen = manifest["sources"]["codecs"][codec_name]
    if str(run) != str(Path(frozen["run"]).resolve(strict=False)):
        raise ValueError(f"{codec_name} run path differs from frozen manifest")
    config_hash = sha256(run / "config.yaml")
    checkpoint_hash = sha256(run / "state.pth")
    if config_hash != frozen["config_sha256"]:
        raise ValueError(f"{codec_name} config hash mismatch")
    if checkpoint_hash != frozen["checkpoint_sha256"]:
        raise ValueError(f"{codec_name} checkpoint hash mismatch")

    config = OmegaConf.load(run / "config.yaml")
    fields = tuple(str(field) for field in config.dataset.fields)
    dimensions = tuple(str(value) for value in config.dataset.dimensions)
    mean = np.asarray(config.dataset.stats.mean, dtype=np.float64)
    std = np.asarray(config.dataset.stats.std, dtype=np.float64)
    transform = {
        int(key): str(value)
        for key, value in OmegaConf.to_container(
            config.dataset.transform, resolve=True
        ).items()
    }
    periodic = tuple(bool(value) for value in config.ae.periodic)
    if fields != C5_FIELDS or dimensions != ("x", "y", "z"):
        raise ValueError(f"{codec_name} field or axis convention changed")
    np.testing.assert_array_equal(mean, LEGACY_MEAN)
    np.testing.assert_array_equal(std, LEGACY_STD)
    if transform != {0: "log_eps"} or periodic != (False, False, True):
        raise ValueError(f"{codec_name} preprocessing or periodicity changed")
    if (
        int(config.ae.pix_channels) != 5
        or int(config.ae.lat_channels) != 64
        or int(config.ae.spatial) != 3
        or float(config.ae.latent_noise) != 0.0
    ):
        raise ValueError(f"{codec_name} architecture identity changed")
    return config, {
        "name": codec_name,
        "run": str(run),
        "config_sha256": config_hash,
        "checkpoint_sha256": checkpoint_hash,
        "fields": list(fields),
        "dimensions": list(dimensions),
        "legacy_mean": mean.tolist(),
        "legacy_std": std.tolist(),
        "transform": {str(key): value for key, value in transform.items()},
        "periodic": list(periodic),
        "latent_noise": float(config.ae.latent_noise),
        "latent_shape": list(CODEC_SPECS[codec_name]["latent_shape"]),
        "lineage": CODEC_SPECS[codec_name]["lineage"],
    }


def _load_codec(
    codec_name: str,
    run: Path,
    config: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    model = get_autoencoder(**config.ae)
    state = torch.load(run / "state.pth", weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.to(device=device, dtype=torch.float32)
    model.requires_grad_(False)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.inference_mode():
        probe = torch.zeros(
            (1, len(C5_FIELDS), *CODEC_SHAPE),
            dtype=torch.float32,
            device=device,
        )
        latent = model.encode(probe)
        expected_latent = (1, *CODEC_SPECS[codec_name]["latent_shape"])
        if tuple(latent.shape) != expected_latent:
            raise ValueError(
                f"{codec_name} latent shape {tuple(latent.shape)} != {expected_latent}"
            )
        reconstruction = model.decode(latent, noisy=False)
        if tuple(reconstruction.shape) != tuple(probe.shape):
            raise ValueError(f"{codec_name} decode shape changed")
    del probe, latent, reconstruction
    return model, parameter_count


def _h5_array(handle: h5py.File, name: str) -> np.ndarray:
    dataset = handle[name]
    values = np.asarray(dataset[...], dtype=np.float64)
    for attribute in ("_FillValue", "missing_value"):
        if attribute in dataset.attrs:
            fill = float(np.asarray(dataset.attrs[attribute]).reshape(-1)[0])
            values = np.where(values == fill, np.nan, values)
    return values


def _load_geometry(
    path: Path,
    geometry_manifest: dict[str, Any],
):
    grid = geometry_manifest["grid"]
    offset = int(grid["model_x_to_grid_x_offset"])
    n_x, n_y = map(int, grid["model_shape"])
    model_slice = slice(offset, offset + n_x)
    topology_record = grid["topology"]
    topology = SingleNullTopology(
        separatrix_x_index=int(topology_record["model_first_sol_x"]),
        core_lower_y=int(topology_record["core_y_inclusive"][0]),
        core_upper_y=int(topology_record["core_y_inclusive"][1]),
        pfr_lower_y=int(topology_record["inner_leg_y_inclusive"][1]),
        pfr_upper_y=int(topology_record["outer_leg_y_inclusive"][0]),
    )
    with h5py.File(path, "r") as handle:
        arrays = {
            name: _h5_array(handle, source)[model_slice]
            for name, source in {
                "jacobian": "J",
                "g11": "g11",
                "g23": "g23",
                "bxy": "Bxy",
                "z_shift": "zShift",
                "dy": "dy",
                "penalty_mask": "penalty_mask",
            }.items()
        }
        shift_angle = _h5_array(handle, "ShiftAngle")[model_slice]
        global_sep_face = int(topology_record["ixseps1"])
        separatrix_radius = _h5_array(handle, "Rxy_xlow")[global_sep_face]
    geometry = build_codec_transport_geometry(
        **arrays,
        shift_angle=shift_angle,
        separatrix_face_major_radius=separatrix_radius,
        dz=toroidal_wedge_spacing(int(grid["native_z_cells"]), zperiod=ZPERIOD),
        topology=topology,
    )
    if geometry.jacobian.shape != (n_x, n_y):
        raise ValueError("geometry crop differs from frozen model shape")
    if int(np.sum(geometry.separatrix_face_mask)) != 16:
        raise ValueError("confined separatrix surface does not contain 16 rows")
    return geometry


def _read_fields(
    trajectory: VirtualWellTrajectory,
    fields: tuple[str, ...],
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    return {
        field: trajectory.read_field(
            field,
            start,
            stop,
            chunk_frames=stop - start,
        )
        for field in fields
    }


def _legacy_state(fields: dict[str, np.ndarray]) -> np.ndarray:
    state = np.stack([fields[field] for field in C5_FIELDS], axis=1).astype(
        np.float32, copy=False
    )
    if state.shape[2:] != CODEC_SHAPE:
        raise ValueError(f"legacy C5T state has unexpected shape {state.shape}")
    if not np.all(np.isfinite(state)) or np.any(state[:, 0] + 1e-6 <= 0.0):
        raise ValueError("legacy C5T input is invalid")
    return state


def _decode_codec(
    model: torch.nn.Module,
    linear_input: np.ndarray,
    *,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[np.ndarray, float]:
    started = time.monotonic()
    with torch.inference_mode():
        input_tensor = torch.from_numpy(linear_input).to(
            device=device, dtype=torch.float32
        )
        model_input = field_preprocess(
            input_tensor.clone(),
            mean=mean,
            std=std,
            transform={0: "log_eps"},
            dim=1,
        )
        latent = model.encode(model_input)
        model_reconstruction = model.decode(latent, noisy=False)
        linear_reconstruction = field_postprocess(
            model_reconstruction.clone(),
            mean=mean,
            std=std,
            transform={0: "log_eps"},
            dim=1,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result = linear_reconstruction.cpu().numpy()
    elapsed = time.monotonic() - started
    if result.shape != linear_input.shape or not np.all(np.isfinite(result)):
        raise ValueError("codec reconstruction is invalid")
    return result, elapsed


def _update_nonpositive(
    accumulator: dict[str, dict[str, int | float]],
    state: dict[str, np.ndarray],
) -> None:
    for field in ("Ne", "Pe", "Pi"):
        values = np.asarray(state[field], dtype=np.float64)
        record = accumulator[field]
        record["sample_count"] = int(record["sample_count"]) + values.size
        record["nonpositive_count"] = int(record["nonpositive_count"]) + int(
            np.count_nonzero(values <= 0.0)
        )
        record["minimum"] = min(float(record["minimum"]), float(np.min(values)))


def _finalize_nonpositive(
    accumulator: dict[str, dict[str, int | float]],
) -> dict[str, Any]:
    result = {}
    for field, record in accumulator.items():
        total = int(record["sample_count"])
        count = int(record["nonpositive_count"])
        result[field] = {
            "sample_count": total,
            "nonpositive_count": count,
            "nonpositive_fraction": count / total,
            "minimum": float(record["minimum"]),
        }
    return result


def _shared_truth_digest(summary: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in ("P0", "P1", "P2"):
        for quantity in TRANSPORT_QUANTITIES:
            values = np.ascontiguousarray(
                summary["surface_series_normalized"][path][quantity],
                dtype=np.float64,
            )
            digest.update(f"{path}:{quantity}".encode("utf-8"))
            digest.update(values.tobytes())
    for comparison in ("P0_vs_P1_state_gap", "P1_vs_P2_input_roundtrip"):
        record = summary["comparisons"][comparison]
        canonical = json.dumps(
            _json_safe(record), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(canonical)
    return digest.hexdigest()


def _attach_si_series(summary: dict[str, Any]) -> None:
    scales = hermes_transport_scales()
    converted = {}
    for path, quantities in summary["surface_series_normalized"].items():
        converted[path] = {}
        for quantity, values in quantities.items():
            scale = (
                scales.particle_rate_scale_per_s
                if quantity == "particle"
                else scales.pressure_flow_scale_w
            )
            converted[path][quantity] = np.asarray(values) * scale
    summary["surface_series_si"] = converted
    summary["surface_series_si_units"] = {
        "particle": "s^-1",
        "electron_internal_energy": "W",
        "ion_internal_energy": "W",
        "total_internal_energy": "W",
    }


def _prior_status(prior: dict[str, Any], codec_name: str) -> str:
    status = prior["codec_results"][codec_name]["preliminary_gate"][
        "preliminary_status"
    ]
    if status not in {"pass", "fail"}:
        raise ValueError(f"invalid prior O1 status for {codec_name}: {status}")
    return status


def main() -> None:
    args = parse_args()
    if args.chunk_frames <= 0 or args.chunk_frames > BLOCK_FRAMES:
        raise ValueError("chunk-frames must lie between 1 and 78")
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    command_file = _resolved_file(args.command_file)
    manifest_path = _resolved_file(args.manifest)
    geometry_manifest_path = _resolved_file(args.geometry_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry_manifest = json.loads(
        geometry_manifest_path.read_text(encoding="utf-8")
    )
    if (
        manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
    ):
        raise ValueError("O1 transport manifest does not preserve the blind-test lock")
    verified = _verify_sources(args, manifest)
    paths = verified["paths"]
    if geometry_manifest["sources"]["geometry"]["sha256"] != sha256(
        paths["geometry"]
    ):
        raise ValueError("released geometry manifest disagrees with O1 geometry")

    native = VirtualWellTrajectory(
        [paths["native_train"], paths["native_valid"]],
        required_fields=NATIVE_FIELDS,
    )
    legacy = VirtualWellTrajectory(
        [paths["legacy_train"], paths["legacy_valid"]],
        required_fields=C5_FIELDS,
    )
    if native.total_frames != TOTAL_FRAMES or legacy.total_frames != TOTAL_FRAMES:
        raise ValueError("both trajectories must contain exactly 624 frames")
    if tuple(native.spatial_shape) != NATIVE_SHAPE:
        raise ValueError(f"native trajectory shape is {native.spatial_shape}")
    if tuple(legacy.spatial_shape) != CODEC_SHAPE:
        raise ValueError(f"legacy trajectory shape is {legacy.spatial_shape}")
    np.testing.assert_array_equal(native.time, legacy.time)
    np.testing.assert_array_equal(native.axes["x"], legacy.axes["x"])
    np.testing.assert_array_equal(native.axes["y"], legacy.axes["y"])

    geometry = _load_geometry(paths["geometry"], geometry_manifest)
    f8_run = _resolved_run(args.f8_run, "f8")
    z44_run = _resolved_run(args.z44_run, "z44")
    runs = {"f8": f8_run, "z44": z44_run}
    configs: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for codec_name, run in runs.items():
        configs[codec_name], identities[codec_name] = _verify_codec_run(
            codec_name, run, manifest
        )

    prior = json.loads(paths["prior_o1"].read_text(encoding="utf-8"))
    for codec_name in runs:
        expected_status = manifest["sources"]["codecs"][codec_name][
            "preliminary_status"
        ]
        if _prior_status(prior, codec_name) != expected_status:
            raise ValueError(f"prior O1 status mismatch for {codec_name}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    determinism = _configure_determinism()
    models: dict[str, torch.nn.Module] = {}
    parameter_counts: dict[str, int] = {}
    for codec_name, run in runs.items():
        models[codec_name], parameter_counts[codec_name] = _load_codec(
            codec_name, run, configs[codec_name], device
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    mean = torch.as_tensor(LEGACY_MEAN, dtype=torch.float32, device=device)
    std = torch.as_tensor(LEGACY_STD, dtype=torch.float32, device=device)
    overall_accumulators = {
        codec_name: TransportComparisonAccumulator() for codec_name in runs
    }
    block_records: dict[str, list[dict[str, Any]]] = {
        codec_name: [] for codec_name in runs
    }
    input_alignment: dict[str, list[np.ndarray]] = {
        field: [] for field in C5_FIELDS
    }
    nonpositive = {
        codec_name: {
            field: {
                "sample_count": 0,
                "nonpositive_count": 0,
                "minimum": math.inf,
            }
            for field in ("Ne", "Pe", "Pi")
        }
        for codec_name in runs
    }
    timing = {
        "shared_transport_seconds": 0.0,
        "codec_decode_seconds": {codec_name: 0.0 for codec_name in runs},
        "codec_transport_seconds": {codec_name: 0.0 for codec_name in runs},
    }
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    for block_index in range(TOTAL_FRAMES // BLOCK_FRAMES):
        block_start = block_index * BLOCK_FRAMES
        block_stop = block_start + BLOCK_FRAMES
        block_accumulators = {
            codec_name: TransportComparisonAccumulator() for codec_name in runs
        }
        for chunk_start in range(block_start, block_stop, args.chunk_frames):
            chunk_stop = min(chunk_start + args.chunk_frames, block_stop)
            native_fields = _read_fields(
                native, NATIVE_FIELDS, chunk_start, chunk_stop
            )
            legacy_fields = _read_fields(
                legacy, C5_FIELDS, chunk_start, chunk_stop
            )
            codec_input = _legacy_state(legacy_fields)
            downsampled_input = periodic_resample_float32(
                codec_input, NATIVE_SHAPE[-1], axis=-1
            )
            for index, field in enumerate(C5_FIELDS):
                input_alignment[field].append(
                    per_frame_relative_l2(
                        native_fields[field], downsampled_input[:, index]
                    )
                )

            p0 = direct_pressure_transport_state(
                native_fields["Ne"],
                native_fields["Pe"],
                native_fields["Pi"],
                native_fields["phi"],
            )
            p1 = c5t_transport_state(
                native_fields["Ne"],
                native_fields["Te"],
                native_fields["Ti"],
                native_fields["phi"],
            )
            p2 = c5t_transport_state(
                downsampled_input[:, 0],
                downsampled_input[:, 1],
                downsampled_input[:, 2],
                downsampled_input[:, 3],
            )
            shared_started = time.monotonic()
            shared_outputs = {
                "P0": evaluate_transport_state(p0, geometry),
                "P1": evaluate_transport_state(p1, geometry),
                "P2": evaluate_transport_state(p2, geometry),
            }
            timing["shared_transport_seconds"] += (
                time.monotonic() - shared_started
            )

            for codec_name, model in models.items():
                reconstruction, elapsed = _decode_codec(
                    model,
                    codec_input,
                    device=device,
                    mean=mean,
                    std=std,
                )
                timing["codec_decode_seconds"][codec_name] += elapsed
                downsampled_reconstruction = periodic_resample_float32(
                    reconstruction, NATIVE_SHAPE[-1], axis=-1
                )
                reconstructed_state = c5t_transport_state(
                    downsampled_reconstruction[:, 0],
                    downsampled_reconstruction[:, 1],
                    downsampled_reconstruction[:, 2],
                    downsampled_reconstruction[:, 3],
                )
                _update_nonpositive(nonpositive[codec_name], reconstructed_state)
                transport_started = time.monotonic()
                reconstructed_output = evaluate_transport_state(
                    reconstructed_state, geometry
                )
                timing["codec_transport_seconds"][codec_name] += (
                    time.monotonic() - transport_started
                )
                path_outputs = {**shared_outputs, "R": reconstructed_output}
                overall_accumulators[codec_name].update(path_outputs)
                block_accumulators[codec_name].update(path_outputs)
                del (
                    reconstruction,
                    downsampled_reconstruction,
                    reconstructed_state,
                    reconstructed_output,
                    path_outputs,
                )

            del (
                native_fields,
                legacy_fields,
                codec_input,
                downsampled_input,
                p0,
                p1,
                p2,
                shared_outputs,
            )

        for codec_name in runs:
            block_records[codec_name].append(
                {
                    "block_index": block_index,
                    "start_inclusive": block_start,
                    "stop_exclusive": block_stop,
                    "metrics": block_accumulators[codec_name].finalize(),
                }
            )
        print(
            f"completed block {block_index + 1}/8 frames "
            f"[{block_start},{block_stop})",
            flush=True,
        )

    input_alignment_values = {
        field: np.concatenate(parts) for field, parts in input_alignment.items()
    }
    for field, values in input_alignment_values.items():
        if values.shape != (TOTAL_FRAMES,):
            raise ValueError(f"input alignment frame count changed for {field}")
    input_alignment_max = {
        field: float(np.max(values))
        for field, values in input_alignment_values.items()
    }

    codec_results: dict[str, Any] = {}
    truth_digests: dict[str, str] = {}
    for codec_name in runs:
        overall = overall_accumulators[codec_name].finalize()
        _attach_si_series(overall)
        block_summaries = [record["metrics"] for record in block_records[codec_name]]
        gate = build_o1_transport_gate(
            overall=overall,
            temporal_blocks=block_summaries,
            input_field_max_relative_l2=input_alignment_max,
            preliminary_status=_prior_status(prior, codec_name),
            thresholds=manifest["acceptance_gates"],
        )
        truth_digests[codec_name] = _shared_truth_digest(overall)
        codec_results[codec_name] = {
            "identity": identities[codec_name],
            "parameter_count": parameter_counts[codec_name],
            "nonpositive_reconstruction": _finalize_nonpositive(
                nonpositive[codec_name]
            ),
            "overall": overall,
            "temporal_blocks": block_records[codec_name],
            "gate": gate,
            "shared_truth_path_sha256": truth_digests[codec_name],
        }
    if truth_digests["f8"] != truth_digests["z44"]:
        raise ValueError("shared P0/P1/P2 truth paths differ between codec passes")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_memory = int(torch.cuda.max_memory_allocated(device))
        properties = torch.cuda.get_device_properties(device)
        gpu_record: dict[str, Any] | None = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [properties.major, properties.minor],
        }
    else:
        peak_cuda_memory = None
        gpu_record = None
    completed_at = datetime.now(timezone.utc)
    scales = hermes_transport_scales()
    result = {
        "schema_version": "0.1.0",
        "result_type": "phase2_o1_codec_transport_oracle",
        "status": "completed",
        "scope": {
            "run_id": "85604",
            "frames": [0, TOTAL_FRAMES],
            "frame_count": TOTAL_FRAMES,
            "temporal_blocks": 8,
            "block_frames": BLOCK_FRAMES,
            "native_shape": list(NATIVE_SHAPE),
            "codec_shape": list(CODEC_SHAPE),
            "zperiod": ZPERIOD,
            "mode_mapping": "n=5*k",
            "shot_85606_accessed": False,
            "training_performed": False,
            "learning_gate_reopened": False,
        },
        "execution": {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": str(args.slurm_job_id),
            "command_file": str(command_file),
            "command": command_file.read_text(encoding="utf-8").strip(),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": gpu_record,
            "chunk_frames": args.chunk_frames,
            "peak_cuda_memory_bytes": peak_cuda_memory,
            "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "determinism": determinism,
            "timing": timing,
        },
        "provenance": {
            "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "geometry_manifest": {
                "path": str(geometry_manifest_path),
                "sha256": sha256(geometry_manifest_path),
            },
            "sources": verified["records"],
            "lola_package_composite_sha256": manifest["sources"][
                "lola_package_composite_sha256"
            ],
        },
        "alignment": {
            "time_exact": True,
            "x_coordinates_exact": True,
            "y_coordinates_exact": True,
            "per_frame_relative_l2": input_alignment_values,
            "maximum_per_frame_relative_l2": input_alignment_max,
        },
        "geometry": {
            "left_cell_indices": geometry.left_cell_indices,
            "strict_face_row_count": int(np.sum(geometry.strict_face_mask)),
            "separatrix_face_row_count": int(
                np.sum(geometry.separatrix_face_mask)
            ),
            "separatrix_face_left_model_x": (
                geometry.region_masks.separatrix_face_left_cell_index
            ),
            "separatrix_y_inclusive": [
                geometry.region_masks.core_lower_y,
                geometry.region_masks.core_upper_y,
            ],
            "dz": geometry.dz,
            "toroidal_convention": "simulated_one_fifth_wedge",
        },
        "units": {
            "particle_scale_per_s": scales.particle_rate_scale_per_s,
            "pressure_flow_scale_W": scales.pressure_flow_scale_w,
            "internal_energy_factor": 1.5,
        },
        "shared_truth": {
            "single_evaluation_fed_to_both_codec_comparisons": True,
            "bitwise_digest_identical": True,
            "sha256": truth_digests["f8"],
        },
        "codec_results": codec_results,
        "decision": {
            "neither_codec_can_full_pass_because_prior_preliminary_failed": all(
                codec_results[name]["gate"]["prior_preliminary"]["status"]
                == "fail"
                for name in runs
            ),
            "no_automatic_architecture_change": True,
            "no_85606_access_authorized": True,
        },
    }
    _write_json_atomic(output, result)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
