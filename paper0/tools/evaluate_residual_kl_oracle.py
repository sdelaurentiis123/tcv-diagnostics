#!/usr/bin/env python3
"""Evaluate the closed residual-KL representation and static ensemble on 85604."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "paper0-residual-kl-oracle-v1"


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_field_metrics import (  # noqa: E402
    FieldRegionAccumulator,
    b2_region_masks,
    pointwise_ensemble_diagnostics,
)
from tcv_diagnostics.b2_field_scoring import (  # noqa: E402
    B2FieldScoreAccumulator,
)
from tcv_diagnostics.b2_forecast import sampler_seed  # noqa: E402
from tcv_diagnostics.b2_spectral_metrics import B2SpectralAccumulator  # noqa: E402
from tcv_diagnostics.b5_covariance_localization import (  # noqa: E402
    B5_COVARIANCE_TOROIDAL_BANDS,
    CovarianceSummaryAccumulator,
    ToroidalPowerAccumulator,
    TransportCovarianceAccumulator,
    association_summary,
    exact_separatrix_local_contributions,
    field_variogram_score,
)
from tcv_diagnostics.b5_residual_audit import B5_FIELDS  # noqa: E402
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.codec_transport import (  # noqa: E402
    TRANSPORT_QUANTITIES,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from tcv_diagnostics.matched_o1_transport import (  # noqa: E402
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import (  # noqa: E402
    CodecFrameDataset,
    VOLUME_SHAPE,
    load_official_catalog,
)
from tcv_diagnostics.o2_forecast import O2ForecastArtifact  # noqa: E402
from tcv_diagnostics.resampling import periodic_resample_float32  # noqa: E402
from tcv_diagnostics.residual_kl_metrics import (  # noqa: E402
    material_power_ratio_summary,
    paired_relative_l2,
    projection_dependence_distance_summary,
    projection_dependence_pass_summary,
    representation_pass_summary,
    residual_cross_spectral_summary,
    static_covariance_usefulness_summary,
)
from tcv_diagnostics.residual_kl_oracle import (  # noqa: E402
    KL_MASTER_SEED,
    KL_RANK_LADDER,
    SnapshotKLBasis,
    classify_kl_outcome,
    gauge_fixed_residual,
    reconstruct_static_kl_members,
)


EXPECTED_MANIFEST_SHA256 = (
    "9255699ed902b314cdc27b9d252d1df2fcff794866299ca6dc8708d9671bf575"
)
EXPECTED_PROTOCOL_SHA256 = (
    "3e1006e52793e612a0daaf21c67e9da2298fc83bfa542af1be4ba376a6acaff7"
)
EXPECTED_DECISION_SHA256 = (
    "742ed3bbdafca1949baba67af19840e83f9e8c28fb9efd93cdee53beef7969bb"
)
TARGETS = tuple(range(498, 624))
BLOCK_INTERVALS = (
    (498, 519),
    (519, 540),
    (540, 561),
    (561, 582),
    (582, 603),
    (603, 624),
)
SAMPLE_SHAPE = (len(B5_FIELDS), *VOLUME_SHAPE)
FEATURE_COUNT = int(np.prod(SAMPLE_SHAPE))
FEATURE_CHUNK = 16_384
SAMPLE_CHUNK = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--oracle-manifest-sha256", required=True)
    parser.add_argument("--oracle-protocol", type=Path, required=True)
    parser.add_argument("--oracle-protocol-sha256", required=True)
    parser.add_argument("--decision-memo", type=Path, required=True)
    parser.add_argument("--decision-memo-sha256", required=True)
    parser.add_argument("--pretruth-closure", type=Path, required=True)
    parser.add_argument("--pretruth-closure-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--h1-validation-forecast", type=Path, required=True)
    parser.add_argument("--h1-validation-forecast-sha256", required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--b5-localization-result", type=Path, required=True)
    parser.add_argument("--b5-localization-result-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--scratch-directory", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(stage: str, **values: Any) -> None:
    print(
        json.dumps(
            {"utc": _utc_now(), "stage": stage, **values},
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != str(expected_commit):
        raise RuntimeError(f"Paper 0 commit {actual} differs from {expected_commit}")
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def require_rocky9_cpu_only() -> None:
    release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if release.get("ID") != "rocky" or release.get("VERSION_ID", "").split(".")[0] != "9":
        raise RuntimeError("residual-KL evaluation requires Rocky Linux 9")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in ("", "NoDevFiles"):
        raise RuntimeError("residual-KL evaluation is CPU-only")


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    actual = sha256_path(resolved)
    if actual != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def validate_authority(manifest: Mapping[str, Any], args: argparse.Namespace) -> None:
    if (
        args.oracle_manifest_sha256 != EXPECTED_MANIFEST_SHA256
        or args.oracle_protocol_sha256 != EXPECTED_PROTOCOL_SHA256
        or args.decision_memo_sha256 != EXPECTED_DECISION_SHA256
        or manifest.get("protocol_status")
        != "frozen_post_B5_preimplementation_residual_KL_representation_oracle"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("residual-KL evaluation authority differs")
    data = manifest["data"]
    if (
        data.get("validation_targets") != [498, 624]
        or data.get("validation_target_count") != len(TARGETS)
        or data.get("chronological_blocks")
        != [list(item) for item in BLOCK_INTERVALS]
        or data.get("fields") != list(B5_FIELDS)
        or data.get("volume_shape") != list(SAMPLE_SHAPE)
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
        or data.get("guard_frames_read_allowed") is not False
    ):
        raise RuntimeError("residual-KL validation data contract differs")
    if tuple(manifest["KL_basis"]["rank_ladder"]) != KL_RANK_LADDER:
        raise RuntimeError("residual-KL rank ladder differs")
    forbidden = set(manifest.get("forbidden_scope", ()))
    if not {
        "model_checkpoint_loading",
        "model_inference",
        "neural_network_training_or_finetuning",
        "validation_selected_rank",
        "guard_frame_reads",
        "85606_access",
        "O3_fixed_block_forecast",
        "assimilation",
        "diagnostic_ranking",
    }.issubset(forbidden):
        raise RuntimeError("residual-KL evaluation closed scope differs")
    locks = manifest["evidence_locks"]
    expected = {
        "H1_validation_forecast": args.h1_validation_forecast_sha256,
        "native_truth_result": args.native_truth_result_sha256,
        "geometry_manifest": args.geometry_manifest_sha256,
        "geometry": args.geometry_sha256,
        "B5_covariance_localization_result": args.b5_localization_result_sha256,
    }
    for name, digest in expected.items():
        record = locks[name]
        locked = record.get("sha256")
        if locked != digest:
            raise RuntimeError(f"residual-KL evidence digest differs for {name}")


def _validate_pretruth_closure(
    closure_path: Path,
    closure_sha256: str,
    *,
    h1_validation_sha256: str,
) -> dict[str, Any]:
    closure = load_strict_json(closure_path)
    if (
        closure.get("scope") != "residual_KL_pretruth_closure_85604"
        or closure.get("development_run") != "85604"
        or closure.get("held_out_85606_read") is not False
        or closure.get("guard_frames_read") is not False
        or closure.get("validation_truth_read") is not False
        or closure.get("immutable_before_validation_truth_open") is not True
        or closure.get("rank_selected_from_training_eigenvalues_only") is not True
    ):
        raise RuntimeError("residual-KL pretruth closure differs")
    verified = {"closure": {"path": closure_path, "sha256": closure_sha256}}
    for name in (
        "basis",
        "training_summary",
        "seed_bank",
        "compressed_forecast_manifest",
    ):
        record = closure[name]
        path = verify_input(Path(record["path"]), record["sha256"], f"pretruth {name}")
        verified[name] = {"path": path, "sha256": record["sha256"]}
    compressed = load_strict_json(verified["compressed_forecast_manifest"]["path"])
    if (
        compressed.get("scope")
        != "closed_static_Gaussian_KL_compressed_forecast_85604"
        or compressed.get("validation_truth_read") is not False
        or compressed.get("rank") != closure.get("selected_static_rank")
        or compressed.get("H1_validation_forecast", {}).get("sha256")
        != h1_validation_sha256
        or compressed.get("truth_or_absolute_time_or_diagnostics_enter_generation")
        is not False
        or compressed.get("covariance_empirical_mean_added_to_forecast") is not False
    ):
        raise RuntimeError("closed compressed residual-KL forecast differs")
    verified["record"] = closure
    verified["compressed"] = compressed
    return verified


def _block_name(start: int, stop: int) -> str:
    return f"frames_{start}_{stop - 1}"


def _block_index(target: int) -> int:
    matches = [
        index
        for index, (start, stop) in enumerate(BLOCK_INTERVALS)
        if start <= int(target) < stop
    ]
    if len(matches) != 1:
        raise ValueError(f"target {target} is outside frozen blocks")
    return matches[0]


def _region_masks(geometry: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    flattened = b2_region_masks(geometry.region_masks, n_z=VOLUME_SHAPE[-1])
    xy: dict[str, np.ndarray] = {}
    for name, mask in flattened.items():
        volume = np.asarray(mask, dtype=bool).reshape(*VOLUME_SHAPE)
        if not np.all(volume == volume[..., :1]):
            raise RuntimeError(f"geometry region {name} is not toroidally invariant")
        xy[name] = np.asarray(volume[..., 0], dtype=bool)
    return flattened, xy


def _load_validation_truth(catalog: Any) -> np.ndarray:
    dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="validation",
        frames=TARGETS,
        augment=False,
        seed=1701,
    )
    try:
        values = np.stack([dataset[index]["volume"] for index in range(len(dataset))])
    finally:
        dataset.close()
    if values.shape != (len(TARGETS), *SAMPLE_SHAPE):
        raise RuntimeError("residual-KL validation truth shape differs")
    return np.ascontiguousarray(values, dtype=np.float32)


def _decode(values: np.ndarray, normalization: Any) -> np.ndarray:
    standardized = np.asarray(values, dtype=np.float32)
    if standardized.ndim != 5 or standardized.shape[1:] != SAMPLE_SHAPE:
        raise ValueError("decoded residual-KL tensor must be [sample,5,64,32,88]")
    return np.stack(
        [
            normalization.records[field].decode(standardized[:, channel])
            for channel, field in enumerate(B5_FIELDS)
        ],
        axis=1,
    ).astype(np.float64)


def _covariance_record(
    values: np.ndarray,
    *,
    region_masks_xy: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    accumulator = CovarianceSummaryAccumulator(
        region_masks_xy=region_masks_xy,
        volume_shape=VOLUME_SHAPE,
    )
    for start in range(0, values.shape[0], SAMPLE_CHUNK):
        accumulator.update(values[start : start + SAMPLE_CHUNK])
    return accumulator.finalize()


def _toroidal_record_allow_zero(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    if np.any(array):
        accumulator = ToroidalPowerAccumulator(volume_shape=VOLUME_SHAPE)
        accumulator.update(array)
        return accumulator.finalize()[0]
    fields = {}
    for field in B5_FIELDS:
        fields[field] = {
            "total_mean_parseval_power_density": 0.0,
            "bands": {
                label: {
                    "stored_k_inclusive": [low, 44 if high is None else high],
                    "full_torus_n_inclusive": [5 * low, 220 if high is None else 5 * high],
                    "mean_parseval_power_density": 0.0,
                    "power_fraction": None,
                }
                for label, (low, high) in B5_COVARIANCE_TOROIDAL_BANDS.items()
            },
        }
    return {
        "sample_count": int(array.shape[0]),
        "stored_toroidal_cells": VOLUME_SHAPE[-1],
        "stored_k_maximum": 44,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "fields": fields,
    }


def _validation_variance_capture(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(reconstruction, dtype=np.float64)
    if observed.shape != candidate.shape:
        raise ValueError("validation variance-capture shapes differ")

    def one(first: np.ndarray, second: np.ndarray) -> float:
        denominator = float(np.sum(first * first, dtype=np.float64))
        return 1.0 - float(np.sum((first - second) ** 2, dtype=np.float64)) / denominator

    return {
        "total": one(observed, candidate),
        "fields": {
            field: one(observed[:, channel], candidate[:, channel])
            for channel, field in enumerate(B5_FIELDS)
        },
    }


def _residual_error_summary(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(reconstruction, dtype=np.float64)
    fields = {}
    for channel, field in enumerate(B5_FIELDS):
        error = candidate[:, channel] - observed[:, channel]
        scale = math.sqrt(float(np.mean(observed[:, channel] ** 2)))
        rmse = math.sqrt(float(np.mean(error * error)))
        fields[field] = {
            "RMSE": rmse,
            "MAE": float(np.mean(np.abs(error))),
            "relative_RMS_to_validation_H1_residual_fluctuation": rmse / scale,
        }
    return {"fields": fields}


def _transport_evaluated_from_model88(physical: np.ndarray, geometry: Any) -> dict[str, Any]:
    values = np.asarray(physical, dtype=np.float64)
    native = periodic_resample_float32(values[:, :4], 81, axis=-1).astype(np.float64)
    return evaluate_transport_state(
        direct_pressure_transport_state(
            native[:, 0], native[:, 1], native[:, 2], native[:, 3]
        ),
        geometry,
    )


def _transport_truth(native: Mapping[str, np.ndarray], geometry: Any) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
    evaluated = evaluate_transport_state(
        direct_pressure_transport_state(
            native["Ne"], native["Pe"], native["Pi"], native["phi"]
        ),
        geometry,
    )
    local, closure = exact_separatrix_local_contributions(
        evaluated,
        strict_face_mask=geometry.strict_face_mask,
        separatrix_face_mask=geometry.separatrix_face_mask,
    )
    return evaluated, local, closure


def _deterministic_transport_metrics(
    *,
    standardized_prediction: np.ndarray,
    catalog: Any,
    geometry: Any,
    truth_evaluated: Mapping[str, Any],
    truth_local: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], float]:
    strict = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
    exact = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
    maximum_closure = 0.0
    for start in range(0, standardized_prediction.shape[0], SAMPLE_CHUNK):
        stop = min(start + SAMPLE_CHUNK, standardized_prediction.shape[0])
        physical = _decode(standardized_prediction[start:stop], catalog.normalization)
        evaluated = _transport_evaluated_from_model88(physical, geometry)
        local, closure = exact_separatrix_local_contributions(
            evaluated,
            strict_face_mask=geometry.strict_face_mask,
            separatrix_face_mask=geometry.separatrix_face_mask,
        )
        maximum_closure = max(maximum_closure, closure)
        for quantity in TRANSPORT_QUANTITIES:
            strict[quantity].append(
                np.asarray(
                    evaluated[quantity]["strict_face_contributions"],
                    dtype=np.float64,
                )
            )
            exact[quantity].append(np.asarray(local[quantity], dtype=np.float64))
    strict_array = {name: np.concatenate(values) for name, values in strict.items()}
    exact_array = {name: np.concatenate(values) for name, values in exact.items()}
    metrics: dict[str, Any] = {}
    for quantity in TRANSPORT_QUANTITIES:
        truth_strict = np.asarray(
            truth_evaluated[quantity]["strict_face_contributions"],
            dtype=np.float64,
        )
        metrics[quantity] = {
            "strict_face_relative_L2": paired_relative_l2(
                strict_array[quantity], truth_strict
            ),
            "exact_separatrix_relative_L2": paired_relative_l2(
                exact_array[quantity], truth_local[quantity]
            ),
            "strict_face_mean_prediction": float(np.mean(strict_array[quantity])),
            "strict_face_mean_truth": float(np.mean(truth_strict)),
            "exact_separatrix_mean_prediction": float(np.mean(exact_array[quantity])),
            "exact_separatrix_mean_truth": float(np.mean(truth_local[quantity])),
            "chronological_blocks": {},
        }
        for block_start, block_stop in BLOCK_INTERVALS:
            selected = slice(block_start - 498, block_stop - 498)
            metrics[quantity]["chronological_blocks"][
                _block_name(block_start, block_stop)
            ] = {
                "strict_face_relative_L2": paired_relative_l2(
                    strict_array[quantity][selected], truth_strict[selected]
                ),
                "exact_separatrix_relative_L2": paired_relative_l2(
                    exact_array[quantity][selected], truth_local[quantity][selected]
                ),
            }
    return metrics, strict_array, exact_array, maximum_closure


def _rank_key(rank: int, *, full_rank: int) -> str:
    return f"rank_{rank}" + ("_full_positive_training_rank" if rank == full_rank else "")


def _available_ranks(positive_rank: int) -> list[int]:
    result: list[int] = []
    for value in KL_RANK_LADDER:
        rank = positive_rank if isinstance(value, str) else int(value)
        if rank <= positive_rank and rank not in result:
            result.append(rank)
    return result


def _projection_coefficients(
    fluctuation: np.ndarray,
    modes: h5py.Dataset,
    positive_rank: int,
) -> np.ndarray:
    values = np.asarray(fluctuation, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1)
    if modes.shape != (int(positive_rank), flat.shape[1]):
        raise ValueError("residual-KL projection basis shape differs")
    coefficients = np.zeros((flat.shape[0], positive_rank), dtype=np.float64)
    for start in range(0, flat.shape[1], FEATURE_CHUNK):
        stop = min(start + FEATURE_CHUNK, flat.shape[1])
        coefficients += np.asarray(flat[:, start:stop], dtype=np.float64) @ np.asarray(
            modes[:, start:stop], dtype=np.float64
        ).T
    return coefficients


def _write_projection(
    *,
    coefficients: np.ndarray,
    modes: h5py.Dataset,
    rank: int,
    destination: np.memmap,
) -> None:
    flat = destination.reshape(destination.shape[0], -1)
    if coefficients.shape[0] != flat.shape[0] or modes.shape[1] != flat.shape[1]:
        raise ValueError("residual-KL projection output shape differs")
    if int(rank) < 0 or int(rank) > min(coefficients.shape[1], modes.shape[0]):
        raise ValueError("residual-KL projection rank is unavailable")
    if int(rank) == 0:
        flat[:] = 0.0
        destination.flush()
        return
    for start in range(0, flat.shape[1], FEATURE_CHUNK):
        stop = min(start + FEATURE_CHUNK, flat.shape[1])
        flat[:, start:stop] = np.asarray(
            coefficients[:, :rank] @ np.asarray(modes[:rank, start:stop], dtype=np.float64),
            dtype=np.float32,
        )
    destination.flush()


def _save_figure(fig: Any, output: Path, stem: str) -> list[str]:
    names = []
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
        names.append(path.name)
    plt.close(fig)
    return names


def _write_figures(
    *,
    output: Path,
    training: Mapping[str, Any],
    ranks: Mapping[str, Any],
    static: Mapping[str, Any],
) -> list[str]:
    names: list[str] = []
    rank_values = [record["rank"] for record in ranks.values()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    eigen = np.asarray(training["eigenvalues"], dtype=np.float64)
    axes[0].semilogy(np.arange(1, eigen.size + 1), eigen, color="#264653")
    axes[0].set(xlabel="KL mode rank", ylabel="Training covariance eigenvalue", title="Residual covariance spectrum")
    axes[0].grid(alpha=0.25)
    axes[1].plot(np.arange(1, eigen.size + 1), np.cumsum(eigen) / np.sum(eigen), color="#e76f51")
    axes[1].axhline(0.90, color="black", linestyle="--", label="90% training selector")
    axes[1].axvline(training["selected_static_rank"], color="#2a9d8f", linestyle=":", label=f"static rank {training['selected_static_rank']}")
    axes[1].set(xlabel="KL mode rank", ylabel="Cumulative training variance", ylim=(0, 1.02), title="Training-only rank selection")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    names += _save_figure(fig, output, "residual-kl-eigenspectrum")

    fig, ax = plt.subplots(figsize=(8, 5))
    for field in B5_FIELDS:
        ax.plot(rank_values, [record["variance_capture"]["fields"][field] for record in ranks.values()], marker="o", label=field)
    ax.plot(rank_values, [record["variance_capture"]["total"] for record in ranks.values()], color="black", linewidth=2.2, marker="s", label="all fields")
    ax.axhline(0.80, color="black", linestyle="--", alpha=0.6, label="total gate 0.80")
    ax.axhline(0.60, color="gray", linestyle=":", alpha=0.7, label="per-field gate 0.60")
    ax.set(xlabel="Residual KL rank", ylabel="Validation residual variance captured", title="Truth-projected representation capacity")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    names += _save_figure(fig, output, "residual-kl-reconstruction")

    heat = []
    labels = []
    for field in B5_FIELDS:
        for band in ("k1_3", "k4_5", "k6_7"):
            labels.append(f"{field} {band}")
            heat.append([
                record["material_power"]["fields"][field][band]["power_ratio"] or 0.0
                for record in ranks.values()
            ])
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(np.asarray(heat), aspect="auto", vmin=0.0, vmax=1.4, cmap="viridis")
    ax.set_xticks(range(len(rank_values)), [str(value) for value in rank_values])
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Residual KL rank", title="Projected/true residual power (stored k, n=5k)")
    fig.colorbar(image, ax=ax, label="absolute power ratio")
    names += _save_figure(fig, output, "residual-kl-toroidal-power")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for quantity in TRANSPORT_QUANTITIES:
        axes[0].plot(rank_values, [record["transport"][quantity]["strict_face_relative_L2"] for record in ranks.values()], marker="o", label=quantity)
        axes[1].plot(rank_values, [record["transport"][quantity]["exact_separatrix_relative_L2"] for record in ranks.values()], marker="o", label=quantity)
    axes[0].axhline(0.40, color="black", linestyle="--", label="strict-face gate")
    axes[1].axhline(0.30, color="black", linestyle="--", label="separatrix gate")
    axes[0].set(ylabel="Relative L2", xlabel="Residual KL rank", title="Strict-face transport")
    axes[1].set(xlabel="Residual KL rank", title="Exact-separatrix local transport")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    names += _save_figure(fig, output, "residual-kl-transport")

    field_record = static["field_and_marginal_calibration"]["regions"]["eligible_union"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(B5_FIELDS))
    ratios = [field_record["fields"][field]["corrected_spread_skill"]["ratio"] for field in B5_FIELDS]
    ax.bar(x, ratios, color="#457b9d")
    ax.axhspan(0.8, 1.25, color="#2a9d8f", alpha=0.15, label="frozen useful range")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xticks(x, B5_FIELDS)
    ax.set(ylabel="Corrected spread / ensemble-mean RMSE", title=f"Static KL field calibration (rank {static['rank']})")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    names += _save_figure(fig, output, "residual-kl-static-field-calibration")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(TRANSPORT_QUANTITIES))
    width = 0.36
    local = [static["transport_covariance"]["quantities"][q]["covariance_decomposition"]["local_corrected_spread_skill_ratio"] for q in TRANSPORT_QUANTITIES]
    integrated = [static["transport_covariance"]["quantities"][q]["covariance_decomposition"]["integrated_corrected_spread_skill_ratio"] for q in TRANSPORT_QUANTITIES]
    ax.bar(x - width / 2, local, width, label="local exact-separatrix")
    ax.bar(x + width / 2, integrated, width, label="integrated exact-separatrix")
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xticks(x, TRANSPORT_QUANTITIES, rotation=15, ha="right")
    ax.set(ylabel="Corrected spread / error", title="Static KL transport covariance calibration")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    names += _save_figure(fig, output, "residual-kl-static-transport-covariance")
    return names


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"refusing to write empty table {path}")
    fields = list(records[0])
    if any(list(record) != fields for record in records):
        raise ValueError(f"table columns differ for {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    wall_started = time.monotonic()
    verify_checkout(args.paper0_commit)
    require_rocky9_cpu_only()
    manifest_path = verify_input(args.oracle_manifest, args.oracle_manifest_sha256, "oracle manifest")
    protocol_path = verify_input(args.oracle_protocol, args.oracle_protocol_sha256, "oracle protocol")
    decision_path = verify_input(args.decision_memo, args.decision_memo_sha256, "decision memo")
    closure_path = verify_input(args.pretruth_closure, args.pretruth_closure_sha256, "pretruth closure")
    manifest = load_strict_json(manifest_path)
    validate_authority(manifest, args)
    pretruth = _validate_pretruth_closure(
        closure_path,
        args.pretruth_closure_sha256,
        h1_validation_sha256=args.h1_validation_forecast_sha256,
    )
    verified = {
        "manifest": manifest_path,
        "protocol": protocol_path,
        "decision": decision_path,
        "H1_validation": verify_input(args.h1_validation_forecast, args.h1_validation_forecast_sha256, "H1 validation forecast"),
        "native_truth_result": verify_input(args.native_truth_result, args.native_truth_result_sha256, "native truth result"),
        "geometry_manifest": verify_input(args.geometry_manifest, args.geometry_manifest_sha256, "geometry manifest"),
        "geometry": verify_input(args.geometry, args.geometry_sha256, "geometry"),
        "B5_localization": verify_input(args.b5_localization_result, args.b5_localization_result_sha256, "B5 localization result"),
    }
    output = Path(args.output_directory)
    scratch = Path(args.scratch_directory)
    assert_development_path(output)
    assert_development_path(scratch)
    if output.exists() or scratch.exists():
        raise FileExistsError("residual-KL evaluation output or scratch already exists")
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    _progress("verified_closed_pretruth_before_validation", closure_sha256=args.pretruth_closure_sha256)

    catalog = load_official_catalog(args.artifact_root)
    geometry = load_transport_geometry(
        geometry_path=verified["geometry"],
        geometry_manifest=load_strict_json(verified["geometry_manifest"]),
    )
    region_masks_flat, region_masks_xy = _region_masks(geometry)
    native_catalog = NativeTruthCatalog(load_strict_json(verified["native_truth_result"]))
    b5_localization = load_strict_json(verified["B5_localization"])
    systematic_identities = manifest["systematic_L3_identities"]
    if (
        b5_localization.get("status")
        != "completed_without_retraining_or_downstream_opening"
        or b5_localization.get("held_out_85606_read") is not False
        or b5_localization.get("blockwise_L3", {}).get("systematic_identities")
        != systematic_identities
    ):
        raise RuntimeError("frozen B5 localization identity reference differs")

    # This is the first validation-truth read in this process and can occur
    # only after every pretruth file above has been rehashed and accepted.
    validation_truth_opened_utc = _utc_now()
    validation_truth = _load_validation_truth(catalog)
    native_truth = native_catalog.read(498, 624)
    with O2ForecastArtifact(
        verified["H1_validation"],
        expected_sha256=args.h1_validation_forecast_sha256,
        target_frames=TARGETS,
    ) as artifact:
        validation_h1 = artifact.read(0, len(TARGETS))
    validation_truth_completed_utc = _utc_now()

    basis_path = Path(pretruth["basis"]["path"])
    summary = load_strict_json(pretruth["training_summary"]["path"])
    training_covariance = summary.get("training_covariance_reference", {})
    if (
        training_covariance.get("validation_truth_read") is not False
        or training_covariance.get("convention")
        != (
            "gauge_fixed_axisymmetric_bias_removed_and_full_empirical_"
            "training_mean_removed_covariance_matrix_R"
        )
    ):
        raise RuntimeError("training KL covariance reference differs")
    training_record = training_covariance["record"]
    seed_bank = np.load(pretruth["seed_bank"]["path"], allow_pickle=False)
    if seed_bank.shape != (len(TARGETS), 32) or seed_bank.dtype != np.uint64:
        raise RuntimeError("closed residual-KL seed bank differs")
    with h5py.File(basis_path, "r") as basis_handle:
        if (
            not bool(basis_handle.attrs["completed"])
            or bool(basis_handle.attrs["validation_truth_read"])
            or bool(basis_handle.attrs["held_out_85606_read"])
            or bool(basis_handle.attrs["covariance_empirical_mean_added_to_forecast"])
        ):
            raise RuntimeError("closed residual-KL basis attributes differ")
        bias = np.asarray(basis_handle["axisymmetric_bias_field_x_y"], dtype=np.float64)
        eigenvalues = np.asarray(basis_handle["eigenvalues"], dtype=np.float64)
        modes = basis_handle["modes_flattened"]
        positive_rank = int(basis_handle.attrs["positive_rank"])
        if modes.shape != (positive_rank, FEATURE_COUNT) or eigenvalues.shape != (positive_rank,):
            raise RuntimeError("closed residual-KL basis tensor shape differs")
        validation_residual = gauge_fixed_residual(validation_truth, validation_h1)
        validation_fluctuation = np.asarray(
            validation_residual - bias[None, ..., None], dtype=np.float32
        )
        coefficients = _projection_coefficients(validation_fluctuation, modes, positive_rank)

        validation_record, validation_raw = _covariance_record(
            validation_fluctuation,
            region_masks_xy=region_masks_xy,
        )
        validation_blocks: dict[str, Any] = {}
        for start, stop in BLOCK_INTERVALS:
            block_record, _ = _covariance_record(
                validation_fluctuation[start - 498 : stop - 498],
                region_masks_xy=region_masks_xy,
            )
            validation_blocks[_block_name(start, stop)] = block_record
        projection = np.memmap(
            scratch / "validation_projection_float32.dat",
            mode="w+",
            dtype=np.float32,
            shape=(len(TARGETS), *SAMPLE_SHAPE),
        )
        truth_evaluated, truth_local, truth_closure = _transport_truth(native_truth, geometry)
        maximum_transport_closure = truth_closure
        rank_records: dict[str, Any] = {}
        rank_rows: list[dict[str, Any]] = []
        previous_total_capture = -math.inf
        rank_zero_reference_maximum_difference: float | None = None
        for rank in _available_ranks(positive_rank):
            _write_projection(
                coefficients=coefficients,
                modes=modes,
                rank=rank,
                destination=projection,
            )
            candidate = np.asarray(projection, dtype=np.float32)
            variance_capture = _validation_variance_capture(validation_fluctuation, candidate)
            residual_error = _residual_error_summary(validation_fluctuation, candidate)
            toroidal = _toroidal_record_allow_zero(candidate)
            material_power = material_power_ratio_summary(
                projection={"toroidal_support": toroidal},
                validation=validation_record,
            )
            cross_spectral = residual_cross_spectral_summary(
                candidate,
                validation_fluctuation,
                eligible_xy_mask=region_masks_xy["eligible_union"],
            )
            projection_blocks: dict[str, Any] = {}
            covariance_record: dict[str, Any] | None = None
            if rank > 0:
                covariance_record, _ = _covariance_record(
                    candidate,
                    region_masks_xy=region_masks_xy,
                )
                for start, stop in BLOCK_INTERVALS:
                    block, _ = _covariance_record(
                        candidate[start - 498 : stop - 498],
                        region_masks_xy=region_masks_xy,
                    )
                    projection_blocks[_block_name(start, stop)] = block
                dependence_distances = projection_dependence_distance_summary(
                    training=training_record,
                    validation=validation_record,
                    projection=covariance_record,
                )
                dependence = projection_dependence_pass_summary(
                    training=training_record,
                    validation_blocks=validation_blocks,
                    projection_blocks=projection_blocks,
                    systematic_identities=systematic_identities,
                )
            else:
                dependence_distances = {
                    "undefined_reason": "rank_zero_has_no_dependence_variance"
                }
                dependence = {
                    "passes": False,
                    "identity_pass_count": 0,
                    "undefined_reason": "rank_zero_has_no_dependence_variance",
                    "chronological_block_count": 6,
                }
            prediction = np.asarray(
                validation_h1 + bias[None, ..., None] + candidate,
                dtype=np.float32,
            )
            if rank == 0:
                reference = np.asarray(
                    validation_h1 + bias[None, ..., None], dtype=np.float32
                )
                rank_zero_reference_maximum_difference = float(
                    np.max(np.abs(prediction - reference))
                )
                if rank_zero_reference_maximum_difference != 0.0:
                    raise RuntimeError("rank-zero projection differs from H1 plus bias")
            transport, _, _, closure = _deterministic_transport_metrics(
                standardized_prediction=prediction,
                catalog=catalog,
                geometry=geometry,
                truth_evaluated=truth_evaluated,
                truth_local=truth_local,
            )
            maximum_transport_closure = max(maximum_transport_closure, closure)
            gate = representation_pass_summary(
                variance_capture=variance_capture,
                dependence=dependence,
                material_power=material_power,
                transport=transport,
            )
            if variance_capture["total"] + 2e-12 < previous_total_capture:
                raise RuntimeError("nested residual-KL reconstruction is not monotone")
            previous_total_capture = float(variance_capture["total"])
            chronological: dict[str, Any] = {}
            for block_start, block_stop in BLOCK_INTERVALS:
                name = _block_name(block_start, block_stop)
                selected = slice(block_start - 498, block_stop - 498)
                candidate_block = candidate[selected]
                validation_block = validation_fluctuation[selected]
                block_toroidal = _toroidal_record_allow_zero(candidate_block)
                chronological[name] = {
                    "variance_capture": _validation_variance_capture(
                        validation_block, candidate_block
                    ),
                    "residual_error": _residual_error_summary(
                        validation_block, candidate_block
                    ),
                    "dependence": (
                        dependence.get("by_block", {}).get(name)
                        if rank > 0
                        else {
                            "undefined_reason": (
                                "rank_zero_has_no_dependence_variance"
                            )
                        }
                    ),
                    "material_power": material_power_ratio_summary(
                        projection={"toroidal_support": block_toroidal},
                        validation=validation_blocks[name],
                    ),
                    "cross_phase_and_coherence": residual_cross_spectral_summary(
                        candidate_block,
                        validation_block,
                        eligible_xy_mask=region_masks_xy["eligible_union"],
                    ),
                    "transport": {
                        quantity: transport[quantity]["chronological_blocks"][name]
                        for quantity in TRANSPORT_QUANTITIES
                    },
                }
            key = _rank_key(rank, full_rank=positive_rank)
            rank_records[key] = _json_safe(
                {
                    "rank": rank,
                    "oracle_label": "truth_projected_representation_capacity_not_forecast",
                    "variance_capture": variance_capture,
                    "residual_error": residual_error,
                    "dependence_distances_all_axes_fields_and_regions": (
                        dependence_distances
                    ),
                    "dependence": dependence,
                    "material_power": material_power,
                    "cross_phase_and_coherence": cross_spectral,
                    "transport": transport,
                    "chronological_blocks": chronological,
                    "representation_gate": gate,
                    "covariance_summary": covariance_record,
                }
            )
            rank_rows.append(
                {
                    "rank": rank,
                    "total_validation_variance_captured": variance_capture["total"],
                    "dependence_identity_pass_count": dependence["identity_pass_count"],
                    "material_power_in_range_count": material_power["in_range_count"],
                    "strict_face_transport_pass_count": gate["components"]["transport"]["strict_face_pass_count"],
                    "exact_separatrix_transport_pass_count": gate["components"]["transport"]["exact_separatrix_pass_count"],
                    "representation_gate_passed": gate["passes"],
                }
            )
            _progress(
                "completed_projection_rank",
                rank=rank,
                positive_rank=positive_rank,
                representation_gate_passed=bool(gate["passes"]),
            )

        static_rank = int(pretruth["record"]["selected_static_rank"])
        static_modes = np.asarray(modes[:static_rank], dtype=np.float32).reshape(
            static_rank, *SAMPLE_SHAPE
        )
        static_basis = SnapshotKLBasis(
            eigenvalues=eigenvalues[:static_rank],
            modes=static_modes,
            gram=np.empty((0, 0)),
            sample_shape=SAMPLE_SHAPE,
            relative_threshold=float(summary["relative_positive_eigenvalue_threshold"]),
            maximum_orthonormality_error=float(summary["maximum_stored_mode_orthonormality_error"]),
            full_rank_training_relative_rms=float(summary["full_rank_covariance_centered_training_relative_rms"]),
            minimum_gram_eigenvalue=float(summary["minimum_gram_eigenvalue"]),
        )

    field_accumulator = B2FieldScoreAccumulator(
        model_seed=1701,
        target_frames=TARGETS,
        region_masks=region_masks_flat,
        validation_blocks=tuple(
            tuple(range(start, stop)) for start, stop in BLOCK_INTERVALS
        ),
    )
    spectral_accumulator = B2SpectralAccumulator(
        model_seed=1701,
        target_frames=TARGETS,
        eligible_xy_mask=region_masks_xy["eligible_union"],
    )
    spectral_blocks = [
        B2SpectralAccumulator(
            model_seed=1701,
            target_frames=tuple(range(start, stop)),
            eligible_xy_mask=region_masks_xy["eligible_union"],
        )
        for start, stop in BLOCK_INTERVALS
    ]
    transport_accumulator = TransportCovarianceAccumulator(
        quantities=TRANSPORT_QUANTITIES,
        rows=16,
        n_z=81,
    )
    transport_blocks = [
        TransportCovarianceAccumulator(
            quantities=TRANSPORT_QUANTITIES,
            rows=16,
            n_z=81,
        )
        for _ in BLOCK_INTERVALS
    ]
    strict_accumulators = {
        quantity: FieldRegionAccumulator() for quantity in TRANSPORT_QUANTITIES
    }
    strict_blocks = [
        {quantity: FieldRegionAccumulator() for quantity in TRANSPORT_QUANTITIES}
        for _ in BLOCK_INTERVALS
    ]
    strict_error_sum = {quantity: 0.0 for quantity in TRANSPORT_QUANTITIES}
    strict_truth_sum = {quantity: 0.0 for quantity in TRANSPORT_QUANTITIES}
    field_variogram_rows: list[dict[str, Any]] = []
    field_association_variance: list[float] = []
    field_association_error: list[float] = []
    transport_association_variance = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
    transport_association_error = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
    companion_integrated = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
    truth_integrated = {
        quantity: np.sum(truth_local[quantity], axis=(1, 2), dtype=np.float64)
        for quantity in TRANSPORT_QUANTITIES
    }
    maximum_member_reload_difference = 0.0
    static_generation_seconds = 0.0
    maximum_transport_closure = max(maximum_transport_closure, truth_closure)

    for index, target in enumerate(TARGETS):
        generation_started = time.monotonic()
        members = reconstruct_static_kl_members(
            h1_mean=validation_h1[index],
            axisymmetric_bias=bias,
            basis=static_basis,
            rank=static_rank,
            member_seeds=seed_bank[index],
        ).astype(np.float32)
        static_generation_seconds += time.monotonic() - generation_started
        if index == 0:
            repeated = reconstruct_static_kl_members(
                h1_mean=validation_h1[index],
                axisymmetric_bias=bias,
                basis=static_basis,
                rank=static_rank,
                member_seeds=seed_bank[index],
            ).astype(np.float32)
            maximum_member_reload_difference = float(np.max(np.abs(members - repeated)))
            if maximum_member_reload_difference != 0.0:
                raise RuntimeError("static KL member reload is not exact")
        truth_standardized = validation_truth[index]
        field_accumulator.update(
            target_frame=target,
            standardized_forecast=members[:, None],
            standardized_truth=truth_standardized,
        )
        physical_members = _decode(members, catalog.normalization)
        physical_truth = _decode(
            truth_standardized[None], catalog.normalization
        )[0]
        block_index = _block_index(target)
        spectral_accumulator.update(
            target_frame=target,
            physical_forecast=physical_members,
            physical_truth=physical_truth,
            mirrors=(spectral_blocks[block_index],),
        )
        evaluated = _transport_evaluated_from_model88(physical_members, geometry)
        local_members, closure = exact_separatrix_local_contributions(
            evaluated,
            strict_face_mask=geometry.strict_face_mask,
            separatrix_face_mask=geometry.separatrix_face_mask,
        )
        maximum_transport_closure = max(maximum_transport_closure, closure)
        local_truth_one = {
            quantity: truth_local[quantity][index]
            for quantity in TRANSPORT_QUANTITIES
        }
        transport_accumulator.update(
            target_frame=target,
            forecast=local_members,
            truth=local_truth_one,
        )
        transport_blocks[block_index].update(
            target_frame=target,
            forecast=local_members,
            truth=local_truth_one,
        )
        tie = sampler_seed(1701, target)
        for quantity_index, quantity in enumerate(TRANSPORT_QUANTITIES):
            forecast_strict = np.asarray(
                evaluated[quantity]["strict_face_contributions"], dtype=np.float64
            ).reshape(32, -1)
            truth_strict = np.asarray(
                truth_evaluated[quantity]["strict_face_contributions"][index],
                dtype=np.float64,
            ).reshape(-1)
            diagnostics = pointwise_ensemble_diagnostics(
                forecast_strict,
                truth_strict,
                target_frame=target,
                channel_index=100 + quantity_index,
                spatial_cell_index=np.arange(truth_strict.size, dtype=np.int64),
                tie_seed=tie,
            )
            mask = np.ones(truth_strict.size, dtype=bool)
            strict_accumulators[quantity].update(diagnostics, truth_strict, mask)
            strict_blocks[block_index][quantity].update(diagnostics, truth_strict, mask)
            strict_error_sum[quantity] += float(np.sum(diagnostics.error**2))
            strict_truth_sum[quantity] += float(np.sum(truth_strict**2))
            integrated_members = np.sum(local_members[quantity], axis=(1, 2), dtype=np.float64)
            integrated_truth = float(truth_integrated[quantity][index])
            transport_association_variance[quantity].append(float(np.var(integrated_members, ddof=1)))
            transport_association_error[quantity].append(float((np.mean(integrated_members) - integrated_truth) ** 2))
        members_gauge = np.asarray(members, dtype=np.float32).copy()
        phi = members_gauge[:, B5_FIELDS.index("phi")]
        phi -= np.mean(phi, axis=(1, 2, 3), keepdims=True, dtype=np.float64)
        members_gauge[:, B5_FIELDS.index("phi")] = phi
        truth_gauge = np.asarray(truth_standardized, dtype=np.float32).copy()
        truth_gauge[B5_FIELDS.index("phi")] -= np.mean(
            truth_gauge[B5_FIELDS.index("phi")], dtype=np.float64
        )
        field_variogram_rows.append(
            {
                "target_frame": target,
                **field_variogram_score(
                    members_gauge,
                    truth_gauge,
                    region_masks_xy=region_masks_xy,
                ),
            }
        )
        mean_gauge = np.mean(members_gauge, axis=0, dtype=np.float64)
        field_association_variance.append(float(np.mean(np.var(members_gauge, axis=0, ddof=1))))
        field_association_error.append(float(np.mean((mean_gauge - truth_gauge) ** 2)))
        physical_mean = np.mean(physical_members, axis=0, keepdims=True, dtype=np.float64)
        mean_evaluated = _transport_evaluated_from_model88(physical_mean, geometry)
        mean_local, mean_closure = exact_separatrix_local_contributions(
            mean_evaluated,
            strict_face_mask=geometry.strict_face_mask,
            separatrix_face_mask=geometry.separatrix_face_mask,
        )
        maximum_transport_closure = max(maximum_transport_closure, mean_closure)
        for quantity in TRANSPORT_QUANTITIES:
            companion_integrated[quantity].append(float(np.sum(mean_local[quantity])))
        if (index + 1) % 10 == 0 or index + 1 == len(TARGETS):
            _progress(
                "streamed_static_KL_ensemble",
                completed_targets=index + 1,
                total_targets=len(TARGETS),
                rank=static_rank,
            )

    if maximum_transport_closure > 2e-12:
        raise RuntimeError("residual-KL exact-separatrix local sum closure differs")
    field_record = field_accumulator.finalize()
    spectral_record = spectral_accumulator.finalize()
    spectral_block_records = {
        _block_name(start, stop): accumulator.finalize()
        for (start, stop), accumulator in zip(BLOCK_INTERVALS, spectral_blocks)
    }
    transport_record, transport_raw = transport_accumulator.finalize()
    transport_block_records = {
        _block_name(start, stop): accumulator.finalize()[0]
        for (start, stop), accumulator in zip(BLOCK_INTERVALS, transport_blocks)
    }
    strict_record = {}
    for quantity in TRANSPORT_QUANTITIES:
        strict_record[quantity] = {
            "probabilistic_metrics": strict_accumulators[quantity].finalize(),
            "ensemble_mean_relative_L2": math.sqrt(
                strict_error_sum[quantity] / strict_truth_sum[quantity]
            ),
            "chronological_blocks": {
                _block_name(start, stop): strict_blocks[index][quantity].finalize()
                for index, (start, stop) in enumerate(BLOCK_INTERVALS)
            },
        }
    field_variogram_aggregate = {
        region: float(np.mean([row[region] for row in field_variogram_rows]))
        for region in field_variogram_rows[0]
        if region != "target_frame"
    }
    field_variogram_blocks = {}
    for start, stop in BLOCK_INTERVALS:
        selected = [row for row in field_variogram_rows if start <= row["target_frame"] < stop]
        field_variogram_blocks[_block_name(start, stop)] = {
            region: float(np.mean([row[region] for row in selected]))
            for region in field_variogram_aggregate
        }
    associations = {
        "field_global_equal_scalar_weight": association_summary(
            field_association_variance, field_association_error
        ),
        "integrated_transport": {
            quantity: association_summary(
                transport_association_variance[quantity],
                transport_association_error[quantity],
            )
            for quantity in TRANSPORT_QUANTITIES
        },
        "chronological_blocks": {},
    }
    for start, stop in BLOCK_INTERVALS:
        name = _block_name(start, stop)
        selected = slice(start - 498, stop - 498)
        associations["chronological_blocks"][name] = {
            "field_global_equal_scalar_weight": association_summary(
                field_association_variance[selected],
                field_association_error[selected],
            ),
            "integrated_transport": {
                quantity: association_summary(
                    transport_association_variance[quantity][selected],
                    transport_association_error[quantity][selected],
                )
                for quantity in TRANSPORT_QUANTITIES
            },
        }
    companion = {
        quantity: {
            "transport_of_ensemble_mean_fields_relative_L2": paired_relative_l2(
                np.asarray(companion_integrated[quantity]), truth_integrated[quantity]
            ),
            "memberwise_transport_mean_is_primary": True,
            "chronological_blocks": {
                _block_name(start, stop): paired_relative_l2(
                    np.asarray(companion_integrated[quantity])[
                        start - 498 : stop - 498
                    ],
                    truth_integrated[quantity][start - 498 : stop - 498],
                )
                for start, stop in BLOCK_INTERVALS
            },
        }
        for quantity in TRANSPORT_QUANTITIES
    }
    aggregate_field = field_record["regions"]["eligible_union"]["aggregate"]
    usefulness = static_covariance_usefulness_summary(
        field_corrected_spread_skill=float(
            aggregate_field["equal_channel_corrected_spread_skill_ratio"]
        ),
        transport_covariance_quantities=transport_record["quantities"],
        finite_noncollapsed_members=bool(aggregate_field["all_fields_nonzero_spread"]),
    )
    passing_ranks = [
        record["rank"]
        for record in rank_records.values()
        if record["representation_gate"]["passes"]
    ]
    minimum_passing_rank = min(passing_ranks) if passing_ranks else None
    outcome = classify_kl_outcome(
        minimum_passing_rank=minimum_passing_rank,
        full_positive_rank=positive_rank,
        tier_b_useful=bool(usefulness["passes"]),
    )
    static_record = _json_safe(
        {
            "rank": static_rank,
            "label": "condition_independent_static_Gaussian_KL_baseline",
            "forecast_closed_before_validation_truth": True,
            "field_and_marginal_calibration": field_record,
            "spectral_and_cross_field": spectral_record,
            "spectral_chronological_blocks": spectral_block_records,
            "transport_covariance": transport_record,
            "transport_chronological_blocks": transport_block_records,
            "strict_face_transport": strict_record,
            "field_variogram": {
                "aggregate_region_mean": field_variogram_aggregate,
                "chronological_blocks": field_variogram_blocks,
                "used_as_training_loss": False,
            },
            "spread_error_association": associations,
            "transport_of_ensemble_mean_fields_companion": companion,
            "usefulness_gate": usefulness,
            "maximum_member_reload_difference": maximum_member_reload_difference,
            "generation_cost": {
                "parameter_count": 0,
                "checkpoint_loaded": False,
                "model_function_evaluations": 0,
                "target_count": len(TARGETS),
                "ensemble_size": 32,
                "generated_member_volumes": len(TARGETS) * 32,
                "measured_reconstruction_seconds": static_generation_seconds,
                "seconds_per_target_M32": (
                    static_generation_seconds / len(TARGETS)
                ),
                "seconds_per_member_volume": (
                    static_generation_seconds / (len(TARGETS) * 32)
                ),
                "full_training_basis_file_bytes": basis_path.stat().st_size,
                "static_mode_slice_logical_bytes_float32": (
                    static_rank * FEATURE_COUNT * np.dtype(np.float32).itemsize
                ),
                "static_eigenvalue_logical_bytes_float64": (
                    static_rank * np.dtype(np.float64).itemsize
                ),
                "seed_bank_bytes": seed_bank.nbytes,
            },
        }
    )
    training_for_figures = dict(summary)
    training_for_figures["eigenvalues"] = eigenvalues.tolist()
    training_for_figures["selected_static_rank"] = static_rank
    figure_names = _write_figures(
        output=output,
        training=training_for_figures,
        ranks=rank_records,
        static=static_record,
    )
    rank_table = output / "tier_A_rank_summary.csv"
    _write_csv(rank_table, rank_rows)
    static_table_rows = []
    for quantity in TRANSPORT_QUANTITIES:
        covariance = transport_record["quantities"][quantity]["covariance_decomposition"]
        static_table_rows.append(
            {
                "quantity": quantity,
                "local_corrected_spread_skill": covariance["local_corrected_spread_skill_ratio"],
                "integrated_corrected_spread_skill": covariance["integrated_corrected_spread_skill_ratio"],
                "coherence_multiplier_ratio": covariance["ensemble_to_error_coherence_multiplier_ratio"],
                "scalar_counterfactual_local": covariance["counterfactual_local_spread_skill_after_same_factor"],
                "usefulness_passed": usefulness["transport_quantities"][quantity]["passes"],
            }
        )
    static_table = output / "tier_B_transport_covariance.csv"
    _write_csv(static_table, static_table_rows)

    raw_path = output / "raw_sufficient_statistics.npz"
    raw_arrays: dict[str, np.ndarray] = {
        "tier_A_validation_projection_coefficients": coefficients,
        "tier_B_seed_bank": seed_bank,
        "tier_B_field_predicted_variance": np.asarray(field_association_variance),
        "tier_B_field_squared_error": np.asarray(field_association_error),
    }
    for name, values in validation_raw.items():
        raw_arrays[f"validation_residual__{name}"] = values
    for name, values in transport_raw.items():
        raw_arrays[f"tier_B_transport__{name}"] = values
    for quantity in TRANSPORT_QUANTITIES:
        raw_arrays[f"tier_B_{quantity}_integrated_predicted_variance"] = np.asarray(
            transport_association_variance[quantity]
        )
        raw_arrays[f"tier_B_{quantity}_integrated_squared_error"] = np.asarray(
            transport_association_error[quantity]
        )
    with raw_path.open("xb") as handle:
        np.savez_compressed(handle, **raw_arrays)

    scientific = {
        "schema_version": 1,
        "scope": "residual_KL_representation_oracle_and_static_covariance_85604",
        "status": "completed_without_model_training_or_downstream_opening",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "validation_truth_opened_only_after_pretruth_closure_verified": True,
        "validation_truth_opened_utc": validation_truth_opened_utc,
        "validation_truth_completed_utc": validation_truth_completed_utc,
        "pretruth_closure": {
            "path": str(closure_path),
            "sha256": args.pretruth_closure_sha256,
        },
        "fields": list(B5_FIELDS),
        "target_frames": [498, 624],
        "target_count": len(TARGETS),
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "training_basis_summary": summary,
        "validation_residual_covariance_reference": validation_record,
        "tier_A_truth_projected_representation_oracle": rank_records,
        "tier_A_minimum_passing_rank": minimum_passing_rank,
        "tier_B_static_Gaussian_KL": static_record,
        "primary_outcome": outcome,
        "maximum_exact_separatrix_relative_sum_closure": maximum_transport_closure,
        "integrity_gates": {
            "nested_projection_total_variance_capture_non_decreasing": True,
            "rank_zero_H1_plus_bias_maximum_absolute_difference": (
                rank_zero_reference_maximum_difference
            ),
            "rank_zero_H1_plus_bias_exact": (
                rank_zero_reference_maximum_difference == 0.0
            ),
            "member_seed_reload_exact": maximum_member_reload_difference == 0.0,
            "pretruth_artifacts_rehashed_before_validation_truth": True,
            "training_covariance_reference_used_centered_KL_matrix_R": True,
        },
        "scientific_boundaries": {
            "tier_A_is_forecast": False,
            "tier_A_uses_current_target_truth_coefficients": True,
            "tier_B_is_condition_independent": True,
            "conditional_covariance_identified": False,
            "checkpoint_loaded": False,
            "model_inference_performed": False,
            "optimizer_or_trainable_parameter_created": False,
            "model_training_performed": False,
            "physics_metric_used_as_training_loss": False,
            "O3_launched": False,
            "O4_launched": False,
            "O5_launched": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "steering_performed": False,
            "held_out_85606_read": False,
        },
    }
    scientific = _json_safe(scientific)
    scientific_path = output / "residual_kl_oracle.json"
    write_strict_json_atomic(scientific_path, scientific)
    result = {
        "schema_version": 1,
        "scope": "residual_KL_representation_oracle_and_static_covariance_85604",
        "status": "completed_without_model_training_or_downstream_opening",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "completed_utc": _utc_now(),
        "wall_seconds": time.monotonic() - wall_started,
        "positive_rank": positive_rank,
        "selected_static_rank": static_rank,
        "tier_A_minimum_passing_rank": minimum_passing_rank,
        "tier_B_static_covariance_useful": bool(usefulness["passes"]),
        "primary_outcome": outcome,
        "rank_zero_H1_plus_bias_maximum_absolute_difference": (
            rank_zero_reference_maximum_difference
        ),
        "scientific_result": {
            "path": str(scientific_path.resolve()),
            "sha256": sha256_path(scientific_path),
        },
        "raw_sufficient_statistics": {
            "path": str(raw_path.resolve()),
            "sha256": sha256_path(raw_path),
            "array_count": len(raw_arrays),
        },
        "tables": {
            "tier_A_rank_summary": {"path": str(rank_table.resolve()), "sha256": sha256_path(rank_table)},
            "tier_B_transport_covariance": {"path": str(static_table.resolve()), "sha256": sha256_path(static_table)},
        },
        "figures": [
            {"path": str((output / name).resolve()), "sha256": sha256_path(output / name)}
            for name in figure_names
        ],
        "checkpoint_loaded": False,
        "model_inference_performed": False,
        "optimizer_or_trainable_parameter_created": False,
        "model_training_performed": False,
        "physics_metric_used_as_training_loss": False,
        "O3_launched": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    indexed = [
        scientific_path,
        raw_path,
        rank_table,
        static_table,
        *(output / name for name in figure_names),
        result_path,
    ]
    artifact_index = output / "artifact_sha256.txt"
    with artifact_index.open("x", encoding="utf-8") as handle:
        for path in sorted(indexed, key=lambda value: value.name):
            handle.write(f"{sha256_path(path)}  {path.resolve()}\n")
    _progress(
        "completed_residual_KL_oracle",
        result_sha256=sha256_path(result_path),
        scientific_sha256=sha256_path(scientific_path),
        positive_rank=positive_rank,
        selected_static_rank=static_rank,
        minimum_passing_rank=minimum_passing_rank,
        static_covariance_useful=bool(usefulness["passes"]),
        primary_outcome=outcome,
    )


if __name__ == "__main__":
    main()
