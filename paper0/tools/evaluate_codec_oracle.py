#!/usr/bin/env python3
"""Run the locked 85604-only Phase 2 O1 codec-reconstruction oracle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_oracle import (  # noqa: E402
    CodecMetricAccumulator,
    build_preliminary_gate,
)
from tcv_diagnostics.data_protocol import C5_FIELDS  # noqa: E402
from tcv_diagnostics.well import VirtualWellTrajectory  # noqa: E402

from lola.autoencoder import get_autoencoder  # noqa: E402
from lola.data import field_postprocess, field_preprocess  # noqa: E402


LEGACY_MEAN = np.asarray([-1.9359, 0.9337, 1.2636, 2.8614, -0.1795])
LEGACY_STD = np.asarray([1.4488, 0.5312, 0.4681, 1.2784, 0.9219])
BLOCK_FRAMES = 78
TOTAL_FRAMES = 624
ZPERIOD = 5
EXPECTED_SPATIAL_SHAPE = (64, 32, 88)
CODEC_SPECS = {
    "f8": {
        "expected_run_name": "w24x2ybf_tcv_c5_dcae_3d_tcv_f8c64",
        "expected_config_sha256": (
            "66509d2b0c9a1aaa03959e0e33691d443f39fa24bbad93a0dbb41e291176e776"
        ),
        "expected_checkpoint_sha256": (
            "9f65dc523b8ee32ea5dd87842b99075de15f9aae86d2e71a5da55bc37091a44e"
        ),
        "expected_latent_shape": (64, 8, 4, 11),
        "lineage": "from_scratch_mae_50x2048",
    },
    "z44": {
        "expected_run_name": "z44c6604191_tcv_c5_dcae_3d_tcv_f8z2c64",
        "expected_config_sha256": (
            "5d868c1cfc5a17ce26c2f6ce86ced50d7b55525c6967c5b599b1074058b67284"
        ),
        "expected_checkpoint_sha256": (
            "095d25f9b6e867103d4cfb946cc9ea8a172a5a6db5b28e5726428c4c57e4979d"
        ),
        "expected_latent_shape": (64, 8, 4, 44),
        "lineage": "non_strict_z22_continuation_mae_plus_increment_12x1024",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--valid-h5", required=True)
    parser.add_argument("--f8-run", required=True)
    parser.add_argument("--z44-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--command-file", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolved_file(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"expected file: {path}")
    return path


def _resolved_run(path_text: str, codec_name: str) -> Path:
    path = Path(path_text).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"expected codec directory: {path}")
    if path.name != CODEC_SPECS[codec_name]["expected_run_name"]:
        raise ValueError(
            f"unexpected {codec_name} run name {path.name}; expected "
            f"{CODEC_SPECS[codec_name]['expected_run_name']}"
        )
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
            json.dump(_json_safe(record), handle, indent=2, sort_keys=True, allow_nan=False)
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
        "dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _config_record(codec_name: str, run: Path) -> tuple[Any, dict[str, Any]]:
    config = OmegaConf.load(run / "config.yaml")
    fields = tuple(str(field) for field in config.dataset.fields)
    dimensions = tuple(str(dimension) for dimension in config.dataset.dimensions)
    mean = np.asarray(config.dataset.stats.mean, dtype=np.float64)
    std = np.asarray(config.dataset.stats.std, dtype=np.float64)
    transform = OmegaConf.to_container(config.dataset.transform, resolve=True)
    transform = {int(key): str(value) for key, value in transform.items()}
    periodic = tuple(bool(value) for value in config.ae.periodic)

    if fields != C5_FIELDS:
        raise ValueError(f"{codec_name} field order {fields} is not C5 {C5_FIELDS}")
    if dimensions != ("x", "y", "z"):
        raise ValueError(f"{codec_name} dimensions are {dimensions}")
    np.testing.assert_array_equal(mean, LEGACY_MEAN)
    np.testing.assert_array_equal(std, LEGACY_STD)
    if transform != {0: "log_eps"}:
        raise ValueError(f"{codec_name} transform is {transform}")
    if periodic != (False, False, True):
        raise ValueError(f"{codec_name} periodic axes are {periodic}")
    if int(config.ae.pix_channels) != 5 or int(config.ae.lat_channels) != 64:
        raise ValueError(f"{codec_name} channel configuration is unexpected")
    if int(config.ae.spatial) != 3 or float(config.ae.latent_noise) != 0.0:
        raise ValueError(f"{codec_name} spatial/noise configuration is unexpected")

    record = {
        "name": codec_name,
        "run": str(run),
        "run_name": run.name,
        "config_sha256": CODEC_SPECS[codec_name]["expected_config_sha256"],
        "checkpoint_sha256": CODEC_SPECS[codec_name][
            "expected_checkpoint_sha256"
        ],
        "checkpoint_target": "state.pth",
        "fields": list(fields),
        "dimensions": list(dimensions),
        "legacy_mean": mean.tolist(),
        "legacy_std": std.tolist(),
        "transform": {str(key): value for key, value in transform.items()},
        "periodic": list(periodic),
        "latent_noise": float(config.ae.latent_noise),
        "expected_latent_shape": list(
            CODEC_SPECS[codec_name]["expected_latent_shape"]
        ),
        "lineage": CODEC_SPECS[codec_name]["lineage"],
    }
    return config, record


def _read_state_chunk(
    trajectory: VirtualWellTrajectory,
    start: int,
    stop: int,
) -> np.ndarray:
    fields = [
        trajectory.read_field(
            field,
            start,
            stop,
            chunk_frames=stop - start,
        )
        for field in C5_FIELDS
    ]
    state = np.stack(fields, axis=1).astype(np.float32, copy=False)
    expected = (stop - start, len(C5_FIELDS), *EXPECTED_SPATIAL_SHAPE)
    if state.shape != expected:
        raise ValueError(f"state chunk has shape {state.shape}, expected {expected}")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"non-finite input in frames [{start},{stop})")
    if np.any(state[:, 0] + 1e-6 <= 0):
        raise ValueError(f"invalid density for log transform in [{start},{stop})")
    return state


def _load_codec(codec_name: str, run: Path, config: Any, device: torch.device):
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
            (1, len(C5_FIELDS), *EXPECTED_SPATIAL_SHAPE),
            dtype=torch.float32,
            device=device,
        )
        latent = model.encode(probe)
        expected = (1, *CODEC_SPECS[codec_name]["expected_latent_shape"])
        if tuple(latent.shape) != expected:
            raise ValueError(
                f"{codec_name} latent shape {tuple(latent.shape)} != {expected}"
            )
        reconstructed = model.decode(latent, noisy=False)
        if tuple(reconstructed.shape) != tuple(probe.shape):
            raise ValueError(
                f"{codec_name} reconstruction shape {tuple(reconstructed.shape)} "
                f"!= {tuple(probe.shape)}"
            )
    del probe, latent, reconstructed
    return model, parameter_count


def _evaluate_codec(
    codec_name: str,
    run: Path,
    trajectory: VirtualWellTrajectory,
    *,
    chunk_frames: int,
    device: torch.device,
) -> dict[str, Any]:
    config, identity = _config_record(codec_name, run)
    model, parameter_count = _load_codec(codec_name, run, config, device)
    mean = torch.as_tensor(LEGACY_MEAN, dtype=torch.float32, device=device)
    std = torch.as_tensor(LEGACY_STD, dtype=torch.float32, device=device)
    transform = {0: "log_eps"}
    block_results: list[dict[str, Any]] = []
    overall_accumulator = CodecMetricAccumulator(
        n_z=EXPECTED_SPATIAL_SHAPE[-1], zperiod=ZPERIOD
    )
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for block_index in range(TOTAL_FRAMES // BLOCK_FRAMES):
        block_start = block_index * BLOCK_FRAMES
        block_stop = block_start + BLOCK_FRAMES
        block_accumulator = CodecMetricAccumulator(
            n_z=EXPECTED_SPATIAL_SHAPE[-1], zperiod=ZPERIOD
        )
        for chunk_start in range(block_start, block_stop, chunk_frames):
            chunk_stop = min(chunk_start + chunk_frames, block_stop)
            linear_truth = _read_state_chunk(trajectory, chunk_start, chunk_stop)
            with torch.inference_mode():
                input_tensor = torch.from_numpy(linear_truth).to(
                    device=device, dtype=torch.float32
                )
                model_truth = field_preprocess(
                    input_tensor.clone(),
                    mean=mean,
                    std=std,
                    transform=transform,
                    dim=1,
                )
                latent = model.encode(model_truth)
                model_reconstruction = model.decode(latent, noisy=False)
                linear_reconstruction = field_postprocess(
                    model_reconstruction.clone(),
                    mean=mean,
                    std=std,
                    transform=transform,
                    dim=1,
                )
                model_truth_array = model_truth.cpu().numpy()
                model_reconstruction_array = model_reconstruction.cpu().numpy()
                linear_reconstruction_array = linear_reconstruction.cpu().numpy()
            block_accumulator.update(
                model_truth_array,
                model_reconstruction_array,
                linear_truth,
                linear_reconstruction_array,
            )
            del (
                input_tensor,
                model_truth,
                latent,
                model_reconstruction,
                linear_reconstruction,
                model_truth_array,
                model_reconstruction_array,
                linear_reconstruction_array,
                linear_truth,
            )
        overall_accumulator.merge(block_accumulator)
        block_results.append(
            {
                "block_index": block_index,
                "start_inclusive": block_start,
                "stop_exclusive": block_stop,
                "metrics": block_accumulator.finalize(),
            }
        )
        print(
            f"[{codec_name}] completed block {block_index + 1}/8 "
            f"frames [{block_start},{block_stop})",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory_bytes = None
    elapsed_seconds = time.monotonic() - started
    overall = overall_accumulator.finalize()
    gate = build_preliminary_gate(
        overall,
        [block["metrics"] for block in block_results],
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "identity": identity,
        "parameter_count": parameter_count,
        "chunk_frames": chunk_frames,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": peak_memory_bytes,
        "overall": overall,
        "temporal_blocks": block_results,
        "preliminary_gate": gate,
    }


def _truth_power(record: dict[str, Any], field: str) -> np.ndarray:
    return np.asarray(
        record["overall"]["toroidal_spectral_curves_linear_coordinates"][field][
            "truth_power"
        ],
        dtype=np.float64,
    )


def _assert_shared_truth(results: dict[str, dict[str, Any]]) -> None:
    reference = results["f8"]
    candidate = results["z44"]
    for field in C5_FIELDS:
        np.testing.assert_array_equal(
            _truth_power(reference, field), _truth_power(candidate, field)
        )
    for block_index in range(8):
        for field in C5_FIELDS:
            left = reference["temporal_blocks"][block_index]["metrics"]
            right = candidate["temporal_blocks"][block_index]["metrics"]
            left_power = np.asarray(
                left["toroidal_spectral_curves_linear_coordinates"][field][
                    "truth_power"
                ]
            )
            right_power = np.asarray(
                right["toroidal_spectral_curves_linear_coordinates"][field][
                    "truth_power"
                ]
            )
            np.testing.assert_array_equal(left_power, right_power)


def _comparison_statement(results: dict[str, dict[str, Any]]) -> str:
    f8 = results["f8"]["preliminary_gate"]["preliminary_status"]
    z44 = results["z44"]["preliminary_gate"]["preliminary_status"]
    if f8 == "pass" and z44 == "pass":
        return "both_preliminary_pass_dynamics_is_next_measured_layer"
    if f8 == "pass":
        return "f8_preliminary_pass_z44_fail_do_not_prefer_larger_codec"
    if z44 == "pass":
        return "only_z44_preliminary_pass_requires_matched_codec_retraining"
    return "both_preliminary_fail_representation_repair_precedes_dynamics"


def main() -> None:
    args = parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("chunk-frames must be positive")
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    train_h5 = _resolved_file(args.train_h5)
    valid_h5 = _resolved_file(args.valid_h5)
    command_file = _resolved_file(args.command_file)
    f8_run = _resolved_run(args.f8_run, "f8")
    z44_run = _resolved_run(args.z44_run, "z44")

    trajectory = VirtualWellTrajectory([train_h5, valid_h5])
    if trajectory.total_frames != TOTAL_FRAMES:
        raise ValueError(
            f"expected {TOTAL_FRAMES} frames, found {trajectory.total_frames}"
        )
    if tuple(trajectory.spatial_shape) != EXPECTED_SPATIAL_SHAPE:
        raise ValueError(
            f"expected spatial shape {EXPECTED_SPATIAL_SHAPE}, "
            f"found {trajectory.spatial_shape}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    determinism = _configure_determinism()
    started_at = datetime.now(timezone.utc)

    codec_results: dict[str, dict[str, Any]] = {}
    for codec_name, run in (("f8", f8_run), ("z44", z44_run)):
        print(f"[{codec_name}] loading {run}", flush=True)
        codec_results[codec_name] = _evaluate_codec(
            codec_name,
            run,
            trajectory,
            chunk_frames=args.chunk_frames,
            device=device,
        )
    _assert_shared_truth(codec_results)

    completed_at = datetime.now(timezone.utc)
    gpu_record: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_record = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [properties.major, properties.minor],
            "current_device": torch.cuda.current_device(),
        }
    result = {
        "schema_version": "0.1.0",
        "result_type": "phase2_o1_codec_reconstruction_oracle",
        "status": "completed",
        "scope": {
            "run_id": "85604",
            "frames": [0, TOTAL_FRAMES],
            "frame_count": TOTAL_FRAMES,
            "temporal_blocks": 8,
            "block_frames": BLOCK_FRAMES,
            "zperiod": ZPERIOD,
            "mode_mapping": "n = 5k",
            "historical_exposure": (
                "all blocks are descriptive within one historically inspected run"
            ),
            "shot_85606_accessed": False,
            "learning_gate_reopened": False,
            "transport_gate": "blocked_pending_authoritative_implementation",
        },
        "execution": {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "elapsed_seconds": (completed_at - started_at).total_seconds(),
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": str(args.slurm_job_id),
            "command_file": str(command_file),
            "command": command_file.read_text(encoding="utf-8").strip(),
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": gpu_record,
            "determinism": determinism,
        },
        "data": {
            "train_h5": str(train_h5),
            "valid_h5": str(valid_h5),
            "storage_semantics": "one chronological 85604 trajectory",
            "input_axes": ["time", "channel", "x", "y", "z"],
            "spatial_shape": list(EXPECTED_SPATIAL_SHAPE),
            "fields": list(C5_FIELDS),
            "preprocessing": "legacy checkpoint coordinates",
            "augmentation": "none",
        },
        "codec_results": codec_results,
        "truth_metrics_identical_between_codec_passes": True,
        "protocol_decision": _comparison_statement(codec_results),
        "claims": {
            "establishes": (
                "deterministic compression behavior on historically exposed 85604"
            ),
            "does_not_establish": [
                "transport fidelity",
                "dynamics skill",
                "probabilistic calibration",
                "cross-shot generalization",
                "a causal latent-resolution ablation",
            ],
        },
    }
    _write_json_atomic(output, result)
    print(f"wrote {output}", flush=True)
    print(f"protocol_decision={result['protocol_decision']}", flush=True)


if __name__ == "__main__":
    main()
