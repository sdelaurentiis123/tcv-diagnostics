#!/usr/bin/env python3
"""Fit and close the 85604 residual-KL basis before validation truth opens."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_field_metrics import b2_region_masks  # noqa: E402
from tcv_diagnostics.b5_covariance_localization import (  # noqa: E402
    CovarianceSummaryAccumulator,
)
from tcv_diagnostics.b5_residual_audit import (  # noqa: E402
    B5_FIELDS,
    axisymmetric_residual_bias,
    cross_field_statistics,
    residual_fluctuation,
)
from tcv_diagnostics.b5_residual_forecast import (  # noqa: E402
    B5TrainingForecastArtifact,
)
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.matched_o1_transport import load_transport_geometry  # noqa: E402
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
from tcv_diagnostics.residual_kl_oracle import (  # noqa: E402
    KL_MASTER_SEED,
    KL_POSITIVE_EIGENVALUE_RELATIVE_THRESHOLD,
    KL_RANK_LADDER,
    diagonalize_snapshot_gram,
    generate_seed_bank,
    select_static_rank,
    snapshot_mode_block,
    streaming_snapshot_gram,
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
TRAINING_TARGETS = tuple(range(2, 432))
VALIDATION_TARGETS = tuple(range(498, 624))
TRAINING_COUNT = len(TRAINING_TARGETS)
VALIDATION_COUNT = len(VALIDATION_TARGETS)
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
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--h1-training-forecast", type=Path, required=True)
    parser.add_argument("--h1-training-forecast-sha256", required=True)
    parser.add_argument("--h1-validation-forecast", type=Path, required=True)
    parser.add_argument("--h1-validation-forecast-sha256", required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--training-audit-sha256", required=True)
    parser.add_argument("--training-raw", type=Path, required=True)
    parser.add_argument("--training-raw-sha256", required=True)
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
        raise RuntimeError("residual-KL pretruth build requires Rocky Linux 9")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in ("", "NoDevFiles"):
        raise RuntimeError("residual-KL pretruth build is CPU-only")


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    actual = sha256_path(resolved)
    if actual != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs: {actual}")
    return resolved


def _resolve_locked_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_authority(manifest: Mapping[str, Any], args: argparse.Namespace) -> None:
    if (
        args.oracle_manifest_sha256 != EXPECTED_MANIFEST_SHA256
        or args.oracle_protocol_sha256 != EXPECTED_PROTOCOL_SHA256
        or args.decision_memo_sha256 != EXPECTED_DECISION_SHA256
        or manifest.get("protocol_status")
        != "frozen_post_B5_preimplementation_residual_KL_representation_oracle"
        or manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("residual-KL pretruth authority differs")
    if manifest.get("decision_memo", {}).get("sha256") != EXPECTED_DECISION_SHA256:
        raise RuntimeError("residual-KL decision lock differs")
    if manifest.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("residual-KL protocol lock differs")
    data = manifest["data"]
    if (
        data.get("fields") != list(B5_FIELDS)
        or data.get("volume_shape") != list(SAMPLE_SHAPE)
        or data.get("training_targets") != [2, 432]
        or data.get("training_target_count") != TRAINING_COUNT
        or data.get("guard_targets") != [432, 496]
        or data.get("validation_targets") != [498, 624]
        or data.get("validation_target_count") != VALIDATION_COUNT
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
        or data.get("guard_frames_read_allowed") is not False
    ):
        raise RuntimeError("residual-KL chronology or axes differ")
    basis = manifest["KL_basis"]
    if (
        basis.get("method") != "method_of_snapshots"
        or basis.get("fit_region") != "training_targets_only"
        or basis.get("matrix_shape") != [TRAINING_COUNT, FEATURE_COUNT]
        or basis.get("compute_dtype") != "float64"
        or basis.get("positive_eigenvalue_relative_threshold")
        != KL_POSITIVE_EIGENVALUE_RELATIVE_THRESHOLD
        or tuple(basis.get("rank_ladder", ())) != KL_RANK_LADDER
        or basis.get("maximum_centered_rank") != TRAINING_COUNT - 1
    ):
        raise RuntimeError("residual-KL basis contract differs")
    forbidden = set(manifest.get("forbidden_scope", ()))
    required_forbidden = {
        "model_checkpoint_loading",
        "model_inference",
        "optimizer_or_trainable_parameter_creation",
        "neural_network_training_or_finetuning",
        "guard_frame_reads",
        "85606_access",
        "O3_fixed_block_forecast",
        "assimilation",
        "diagnostic_ranking",
    }
    if not required_forbidden.issubset(forbidden):
        raise RuntimeError("residual-KL closed scope differs")
    locks = manifest["evidence_locks"]
    checks = {
        "H1_training_forecast": (
            args.h1_training_forecast,
            args.h1_training_forecast_sha256,
        ),
        "H1_validation_forecast": (
            args.h1_validation_forecast,
            args.h1_validation_forecast_sha256,
        ),
        "training_residual_audit": (
            args.training_audit,
            args.training_audit_sha256,
        ),
        "training_residual_sufficient_statistics": (
            args.training_raw,
            args.training_raw_sha256,
        ),
        "geometry_manifest": (
            args.geometry_manifest,
            args.geometry_manifest_sha256,
        ),
        "geometry": (args.geometry, args.geometry_sha256),
    }
    for name, (path, digest) in checks.items():
        record = locks[name]
        if record.get("sha256") != digest:
            raise RuntimeError(f"residual-KL evidence digest differs for {name}")
        if _resolve_locked_path(str(record["path"])).resolve() != Path(path).resolve():
            raise RuntimeError(f"residual-KL evidence path differs for {name}")
    localization = locks["B5_covariance_localization_result"]
    if (
        localization.get("sha256") != args.b5_localization_result_sha256
        or (ROOT / localization["tracked_path"]).resolve()
        != Path(args.b5_localization_result).resolve()
    ):
        raise RuntimeError("residual-KL B5 localization lock differs")
    dataset = locks["model_dataset"]
    if (
        Path(dataset["root"]).resolve() != Path(args.artifact_root).resolve()
        or dataset.get("manifest_sha256")
        != "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18"
        or dataset.get("normalization_sha256")
        != "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7"
    ):
        raise RuntimeError("residual-KL model dataset lock differs")


def _region_masks(geometry: Any) -> dict[str, np.ndarray]:
    flattened = b2_region_masks(geometry.region_masks, n_z=VOLUME_SHAPE[-1])
    result: dict[str, np.ndarray] = {}
    for name, mask in flattened.items():
        volume = np.asarray(mask, dtype=bool).reshape(*VOLUME_SHAPE)
        if not np.all(volume == volume[..., :1]):
            raise RuntimeError(f"geometry region {name} is not toroidally invariant")
        result[name] = np.asarray(volume[..., 0], dtype=bool)
    return result


def _training_covariance_reference(
    centered: np.ndarray,
    *,
    region_masks_xy: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Summarize the exact gauge-fixed, covariance-centered KL fit matrix."""

    accumulator = CovarianceSummaryAccumulator(
        region_masks_xy=region_masks_xy,
        volume_shape=VOLUME_SHAPE,
    )
    for start in range(0, TRAINING_COUNT, SAMPLE_CHUNK):
        stop = min(start + SAMPLE_CHUNK, TRAINING_COUNT)
        accumulator.update(centered[start:stop])
    record, _ = accumulator.finalize()
    return record


def _load_training_residual(
    *,
    catalog: Any,
    forecast_path: Path,
    forecast_sha256: str,
    destination: Path,
) -> np.memmap:
    residual = np.memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(TRAINING_COUNT, *SAMPLE_SHAPE),
    )
    dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="train",
        frames=TRAINING_TARGETS,
        augment=False,
        seed=1701,
    )
    try:
        with B5TrainingForecastArtifact(
            forecast_path,
            expected_sha256=forecast_sha256,
            target_frames=TRAINING_TARGETS,
        ) as artifact:
            for start in range(0, TRAINING_COUNT, SAMPLE_CHUNK):
                stop = min(start + SAMPLE_CHUNK, TRAINING_COUNT)
                truth = np.stack(
                    [dataset[index]["volume"] for index in range(start, stop)]
                ).astype(np.float32)
                h1 = artifact.read(start, stop)
                residual[start:stop] = np.asarray(truth - h1, dtype=np.float32)
                if stop % 40 == 0 or stop == TRAINING_COUNT:
                    _progress(
                        "loaded_training_residual",
                        completed_targets=stop,
                        total_targets=TRAINING_COUNT,
                    )
    finally:
        dataset.close()
    residual.flush()
    return residual


def _legacy_integrity(
    *,
    residual: np.ndarray,
    training_audit: Mapping[str, Any],
    training_raw: Mapping[str, np.ndarray],
    region_masks_xy: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    recomputed_bias = axisymmetric_residual_bias(residual)
    stored_bias = np.asarray(
        training_raw["axisymmetric_residual_bias__field_x_y"], dtype=np.float64
    )
    fluctuation = residual_fluctuation(residual, recomputed_bias)
    recomputed_cross, _ = cross_field_statistics(
        fluctuation,
        region_masks_xy=region_masks_xy,
    )
    cross_maximum = 0.0
    cross_worst = "none"
    for region, stored in training_audit["cross_field"].items():
        actual = np.asarray(
            recomputed_cross[region]["correlation_matrix"], dtype=np.float64
        )
        expected = np.asarray(stored["correlation_matrix"], dtype=np.float64)
        difference = float(np.max(np.abs(actual - expected)))
        if difference > cross_maximum:
            cross_maximum = difference
            cross_worst = str(region)
    bias_maximum = float(np.max(np.abs(recomputed_bias - stored_bias)))
    if cross_maximum > 2e-6 or bias_maximum > 2e-6:
        raise RuntimeError("legacy training residual integrity anchors differ")
    return {
        "passed": True,
        "axisymmetric_bias_maximum_absolute_difference": bias_maximum,
        "cross_field_correlation_maximum_absolute_difference": cross_maximum,
        "cross_field_worst_region": cross_worst,
        "tolerance": 2e-6,
        "legacy_ungauged_only": True,
        "used_as_new_phi_gauge_fixed_covariance_reference": False,
    }


def _gauge_and_center_in_place(residual: np.memmap) -> dict[str, np.ndarray | float]:
    for start in range(0, TRAINING_COUNT, SAMPLE_CHUNK):
        stop = min(start + SAMPLE_CHUNK, TRAINING_COUNT)
        values = np.asarray(residual[start:stop], dtype=np.float64)
        phi = values[:, B5_FIELDS.index("phi")]
        phi -= np.mean(phi, axis=(1, 2, 3), keepdims=True, dtype=np.float64)
        values[:, B5_FIELDS.index("phi")] = phi
        residual[start:stop] = np.asarray(values, dtype=np.float32)
    residual.flush()
    bias_sum = np.zeros(SAMPLE_SHAPE[:-1], dtype=np.float64)
    for start in range(0, TRAINING_COUNT, SAMPLE_CHUNK):
        stop = min(start + SAMPLE_CHUNK, TRAINING_COUNT)
        bias_sum += np.sum(
            np.asarray(residual[start:stop], dtype=np.float64),
            axis=(0, 4),
            dtype=np.float64,
        )
    bias = bias_sum / float(TRAINING_COUNT * SAMPLE_SHAPE[-1])
    empirical_sum = np.zeros(SAMPLE_SHAPE, dtype=np.float64)
    for start in range(0, TRAINING_COUNT, SAMPLE_CHUNK):
        stop = min(start + SAMPLE_CHUNK, TRAINING_COUNT)
        fluctuation = (
            np.asarray(residual[start:stop], dtype=np.float64)
            - bias[None, ..., None]
        )
        empirical_sum += np.sum(fluctuation, axis=0, dtype=np.float64)
    empirical = empirical_sum / float(TRAINING_COUNT)
    empirical_z_mean = np.mean(empirical, axis=-1, dtype=np.float64)
    empirical_scale = max(
        float(np.max(np.abs(empirical))), np.finfo(np.float64).tiny
    )
    empirical_z_relative = float(np.max(np.abs(empirical_z_mean))) / empirical_scale
    if empirical_z_relative > 5e-6:
        raise RuntimeError("covariance empirical mean toroidal average differs")
    for start in range(0, TRAINING_COUNT, SAMPLE_CHUNK):
        stop = min(start + SAMPLE_CHUNK, TRAINING_COUNT)
        centered = (
            np.asarray(residual[start:stop], dtype=np.float64)
            - bias[None, ..., None]
            - empirical[None]
        )
        residual[start:stop] = np.asarray(centered, dtype=np.float32)
    residual.flush()
    field_energy = np.zeros(len(B5_FIELDS), dtype=np.float64)
    total_energy = 0.0
    for start in range(0, TRAINING_COUNT, SAMPLE_CHUNK):
        stop = min(start + SAMPLE_CHUNK, TRAINING_COUNT)
        values = np.asarray(residual[start:stop], dtype=np.float64)
        field_energy += np.sum(values * values, axis=(0, 2, 3, 4), dtype=np.float64)
        total_energy += float(np.sum(values * values, dtype=np.float64))
    field_energy /= float(TRAINING_COUNT - 1)
    total_energy /= float(TRAINING_COUNT - 1)
    return {
        "axisymmetric_bias": bias,
        "covariance_empirical_mean": empirical,
        "empirical_mean_toroidal_average_relative_maximum": empirical_z_relative,
        "field_covariance_trace": field_energy,
        "total_covariance_trace": total_energy,
    }


def _effective_ranks(eigenvalues: np.ndarray) -> dict[str, float]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    probabilities = values / np.sum(values)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "participation_ratio": float(np.sum(values) ** 2 / np.sum(values * values)),
        "entropy_effective_rank": math.exp(entropy),
    }


def _fit_and_write_basis(
    *,
    centered: np.memmap,
    centering: Mapping[str, np.ndarray | float],
    output: Path,
    input_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    flat = centered.reshape(TRAINING_COUNT, FEATURE_COUNT)
    gram, maximum_relative_mean = streaming_snapshot_gram(
        flat,
        feature_chunk_size=FEATURE_CHUNK,
    )
    eigensystem = diagonalize_snapshot_gram(
        gram,
        sample_count=TRAINING_COUNT,
        feature_count=FEATURE_COUNT,
        maximum_relative_feature_mean=maximum_relative_mean,
    )
    selector = select_static_rank(eigensystem.eigenvalues)
    rank = eigensystem.positive_rank
    basis_path = output / "training_kl_basis.h5"
    mode_gram = np.zeros((rank, rank), dtype=np.float64)
    coefficients = np.zeros((TRAINING_COUNT, rank), dtype=np.float64)
    mode_field_norm = np.zeros((rank, len(B5_FIELDS)), dtype=np.float64)
    field_width = int(np.prod(SAMPLE_SHAPE[1:]))
    with h5py.File(basis_path, "x") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["scope"] = "training_only_residual_KL_basis_85604"
        handle.attrs["development_run"] = "85604"
        handle.attrs["held_out_85606_read"] = False
        handle.attrs["guard_frames_read"] = False
        handle.attrs["validation_truth_read"] = False
        handle.attrs["checkpoint_loaded"] = False
        handle.attrs["model_inference_performed"] = False
        handle.attrs["training_performed"] = False
        handle.attrs["zperiod"] = 5
        handle.attrs["mode_mapping"] = "n=5k"
        handle.attrs["field_order_json"] = json.dumps(list(B5_FIELDS))
        handle.attrs["sample_shape_json"] = json.dumps(list(SAMPLE_SHAPE))
        handle.attrs["mode_storage_axes_json"] = json.dumps(["rank", "flattened_field_x_y_z"])
        handle.attrs["covariance_empirical_mean_added_to_forecast"] = False
        handle.create_dataset(
            "training_target_frame_index",
            data=np.asarray(TRAINING_TARGETS, dtype=np.int64),
        )
        handle.create_dataset(
            "axisymmetric_bias_field_x_y",
            data=np.asarray(centering["axisymmetric_bias"], dtype=np.float64),
        )
        handle.create_dataset(
            "covariance_empirical_mean_field_x_y_z",
            data=np.asarray(
                centering["covariance_empirical_mean"], dtype=np.float32
            ),
            compression="lzf",
            shuffle=True,
            fletcher32=True,
        )
        handle.create_dataset("snapshot_gram", data=eigensystem.gram)
        handle.create_dataset("eigenvalues", data=eigensystem.eigenvalues)
        handle.create_dataset("snapshot_eigenvectors", data=eigensystem.eigenvectors)
        modes = handle.create_dataset(
            "modes_flattened",
            shape=(rank, FEATURE_COUNT),
            dtype="f4",
            chunks=(min(8, rank), FEATURE_CHUNK),
            compression="lzf",
            shuffle=True,
            fletcher32=True,
        )
        for start in range(0, FEATURE_COUNT, FEATURE_CHUNK):
            stop = min(start + FEATURE_CHUNK, FEATURE_COUNT)
            block = np.asarray(flat[:, start:stop], dtype=np.float64)
            mode_block = snapshot_mode_block(block, eigensystem)
            stored = np.asarray(mode_block, dtype=np.float32)
            modes[:, start:stop] = stored
            stored64 = np.asarray(stored, dtype=np.float64)
            mode_gram += stored64 @ stored64.T
            coefficients += block @ stored64.T
            for channel in range(len(B5_FIELDS)):
                overlap_start = max(start, channel * field_width)
                overlap_stop = min(stop, (channel + 1) * field_width)
                if overlap_start < overlap_stop:
                    local_start = overlap_start - start
                    local_stop = overlap_stop - start
                    selected = stored64[:, local_start:local_stop]
                    mode_field_norm[:, channel] += np.sum(
                        selected * selected, axis=1, dtype=np.float64
                    )
            _progress(
                "constructed_KL_modes",
                completed_features=stop,
                total_features=FEATURE_COUNT,
                positive_rank=rank,
            )
        handle.flush()
        maximum_orthogonality_error = float(
            np.max(np.abs(mode_gram - np.eye(rank)))
        )
        if maximum_orthogonality_error > 2e-5:
            raise RuntimeError("stored residual-KL modes fail orthonormality")
        error_sum = 0.0
        energy_sum = 0.0
        for start in range(0, FEATURE_COUNT, FEATURE_CHUNK):
            stop = min(start + FEATURE_CHUNK, FEATURE_COUNT)
            block = np.asarray(flat[:, start:stop], dtype=np.float64)
            stored = np.asarray(modes[:, start:stop], dtype=np.float64)
            reconstruction = coefficients @ stored
            error_sum += float(
                np.sum((block - reconstruction) ** 2, dtype=np.float64)
            )
            energy_sum += float(np.sum(block * block, dtype=np.float64))
        full_rank_relative_rms = math.sqrt(error_sum / energy_sum)
        if full_rank_relative_rms > 2e-5:
            raise RuntimeError("stored full-rank residual-KL reconstruction fails")
        handle.create_dataset("stored_mode_gram", data=mode_gram)
        handle.create_dataset("training_projection_coefficients", data=coefficients)
        handle.create_dataset("mode_field_squared_norm", data=mode_field_norm)
        handle.attrs["maximum_orthonormality_error"] = maximum_orthogonality_error
        handle.attrs["full_rank_training_relative_rms"] = full_rank_relative_rms
        handle.attrs["positive_rank"] = rank
        handle.attrs["selected_static_rank"] = int(selector["rank"])
        handle.attrs["completed"] = True
        handle.flush()
    basis_sha256 = sha256_path(basis_path)
    eigenvalues = eigensystem.eigenvalues
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    field_trace = np.asarray(centering["field_covariance_trace"], dtype=np.float64)
    cumulative_field = np.cumsum(
        eigenvalues[:, None] * mode_field_norm,
        axis=0,
    ) / field_trace[None]
    requested_ranks = []
    for value in KL_RANK_LADDER:
        resolved = rank if isinstance(value, str) else int(value)
        requested_ranks.append(
            {
                "label": str(value),
                "resolved_rank": resolved,
                "available": resolved <= rank,
                "cumulative_training_variance_fraction": (
                    float(cumulative[resolved - 1]) if 0 < resolved <= rank else 0.0
                ),
                "per_field_cumulative_variance_fraction": {
                    field: (
                        float(cumulative_field[resolved - 1, channel])
                        if 0 < resolved <= rank
                        else 0.0
                    )
                    for channel, field in enumerate(B5_FIELDS)
                },
            }
        )
    empirical = np.asarray(centering["covariance_empirical_mean"], dtype=np.float64)
    summary = {
        "schema_version": 1,
        "scope": "training_only_residual_KL_basis_summary_85604",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "validation_truth_read": False,
        "training_targets": [2, 432],
        "training_target_count": TRAINING_COUNT,
        "fields": list(B5_FIELDS),
        "sample_shape": list(SAMPLE_SHAPE),
        "feature_count": FEATURE_COUNT,
        "positive_rank": rank,
        "maximum_centered_rank": TRAINING_COUNT - 1,
        "relative_positive_eigenvalue_threshold": (
            KL_POSITIVE_EIGENVALUE_RELATIVE_THRESHOLD
        ),
        "minimum_gram_eigenvalue": eigensystem.minimum_gram_eigenvalue,
        "maximum_relative_feature_mean_after_float32_storage": (
            eigensystem.maximum_relative_feature_mean
        ),
        "maximum_stored_mode_orthonormality_error": (
            maximum_orthogonality_error
        ),
        "full_rank_covariance_centered_training_relative_rms": (
            full_rank_relative_rms
        ),
        "training_total_covariance_trace": float(
            centering["total_covariance_trace"]
        ),
        "training_field_covariance_trace": {
            field: float(field_trace[channel])
            for channel, field in enumerate(B5_FIELDS)
        },
        "removed_covariance_empirical_mean": {
            "added_to_forecast_mean": False,
            "total_squared_norm": float(np.sum(empirical * empirical)),
            "per_field_squared_norm": {
                field: float(np.sum(empirical[channel] ** 2))
                for channel, field in enumerate(B5_FIELDS)
            },
            "maximum_relative_toroidal_average": float(
                centering["empirical_mean_toroidal_average_relative_maximum"]
            ),
        },
        "effective_rank": _effective_ranks(eigenvalues),
        "rank_ladder": requested_ranks,
        "static_rank_selection": selector,
        "basis": {
            "path": str(basis_path.resolve()),
            "sha256": basis_sha256,
            "bytes": basis_path.stat().st_size,
            "mode_dtype": "float32",
            "mode_shape": [rank, FEATURE_COUNT],
        },
        "input_sha256": dict(input_sha256),
        "checkpoint_loaded": False,
        "model_inference_performed": False,
        "optimizer_or_trainable_parameter_created": False,
        "training_performed": False,
        "physics_metric_used_as_training_loss": False,
    }
    return summary, selector


def _write_npy_exclusive(path: Path, values: np.ndarray) -> Path:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
    return path


def main() -> None:
    args = parse_args()
    wall_started = time.monotonic()
    verify_checkout(args.paper0_commit)
    require_rocky9_cpu_only()
    manifest_path = verify_input(
        args.oracle_manifest,
        args.oracle_manifest_sha256,
        "residual-KL manifest",
    )
    protocol_path = verify_input(
        args.oracle_protocol,
        args.oracle_protocol_sha256,
        "residual-KL protocol",
    )
    decision_path = verify_input(
        args.decision_memo,
        args.decision_memo_sha256,
        "residual-KL decision memo",
    )
    manifest = load_strict_json(manifest_path)
    validate_authority(manifest, args)
    verified = {
        "manifest": manifest_path,
        "protocol": protocol_path,
        "decision": decision_path,
        "H1_training_forecast": verify_input(
            args.h1_training_forecast,
            args.h1_training_forecast_sha256,
            "H1 training forecast",
        ),
        "H1_validation_forecast": verify_input(
            args.h1_validation_forecast,
            args.h1_validation_forecast_sha256,
            "H1 validation forecast",
        ),
        "training_audit": verify_input(
            args.training_audit,
            args.training_audit_sha256,
            "training residual audit",
        ),
        "training_raw": verify_input(
            args.training_raw,
            args.training_raw_sha256,
            "training residual raw statistics",
        ),
        "geometry_manifest": verify_input(
            args.geometry_manifest,
            args.geometry_manifest_sha256,
            "geometry manifest",
        ),
        "geometry": verify_input(
            args.geometry,
            args.geometry_sha256,
            "geometry",
        ),
        "B5_localization": verify_input(
            args.b5_localization_result,
            args.b5_localization_result_sha256,
            "B5 covariance localization result",
        ),
    }
    output = Path(args.output_directory)
    scratch = Path(args.scratch_directory)
    assert_development_path(output)
    assert_development_path(scratch)
    if output.exists() or scratch.exists():
        raise FileExistsError("residual-KL pretruth output or scratch already exists")
    output.mkdir(parents=True)
    scratch.mkdir(parents=True)
    _progress("verified_pretruth_authority", output=str(output), scratch=str(scratch))

    training_audit = load_strict_json(verified["training_audit"])
    b5_localization = load_strict_json(verified["B5_localization"])
    if (
        training_audit.get("target_frames") != [2, 432]
        or training_audit.get("validation_frames_read") is not False
        or training_audit.get("held_out_85606_read") is not False
        or b5_localization.get("status")
        != "completed_without_retraining_or_downstream_opening"
        or b5_localization.get("held_out_85606_read") is not False
    ):
        raise RuntimeError("residual-KL parent evidence status differs")
    with np.load(verified["training_raw"], allow_pickle=False) as archive:
        training_raw = {name: np.asarray(archive[name]) for name in archive.files}

    catalog = load_official_catalog(args.artifact_root)
    geometry = load_transport_geometry(
        geometry_path=verified["geometry"],
        geometry_manifest=load_strict_json(verified["geometry_manifest"]),
    )
    region_masks_xy = _region_masks(geometry)
    # Verify target order and closed forecast metadata without reading truth.
    with O2ForecastArtifact(
        verified["H1_validation_forecast"],
        expected_sha256=args.h1_validation_forecast_sha256,
        target_frames=VALIDATION_TARGETS,
    ):
        pass
    residual = _load_training_residual(
        catalog=catalog,
        forecast_path=verified["H1_training_forecast"],
        forecast_sha256=args.h1_training_forecast_sha256,
        destination=scratch / "training_residual_float32.dat",
    )
    legacy = _legacy_integrity(
        residual=residual,
        training_audit=training_audit,
        training_raw=training_raw,
        region_masks_xy=region_masks_xy,
    )
    _progress("passed_legacy_training_integrity", **legacy)
    centering = _gauge_and_center_in_place(residual)
    _progress(
        "completed_covariance_centering",
        empirical_mean_toroidal_average_relative_maximum=float(
            centering["empirical_mean_toroidal_average_relative_maximum"]
        ),
    )
    training_covariance_reference = _training_covariance_reference(
        residual,
        region_masks_xy=region_masks_xy,
    )
    _progress(
        "completed_training_covariance_reference",
        convention="gauge_fixed_axisymmetric_bias_removed_covariance_centered_R",
    )
    input_sha256 = {
        "oracle_manifest": args.oracle_manifest_sha256,
        "oracle_protocol": args.oracle_protocol_sha256,
        "decision_memo": args.decision_memo_sha256,
        "H1_training_forecast": args.h1_training_forecast_sha256,
        "H1_validation_forecast": args.h1_validation_forecast_sha256,
        "training_audit": args.training_audit_sha256,
        "training_raw": args.training_raw_sha256,
        "geometry_manifest": args.geometry_manifest_sha256,
        "geometry": args.geometry_sha256,
        "B5_covariance_localization": args.b5_localization_result_sha256,
    }
    summary, selector = _fit_and_write_basis(
        centered=residual,
        centering=centering,
        output=output,
        input_sha256=input_sha256,
    )
    summary["legacy_training_integrity"] = legacy
    summary["training_covariance_reference"] = {
        "convention": (
            "gauge_fixed_axisymmetric_bias_removed_and_full_empirical_"
            "training_mean_removed_covariance_matrix_R"
        ),
        "validation_truth_read": False,
        "record": training_covariance_reference,
    }
    summary_path = output / "training_kl_summary.json"
    write_strict_json_atomic(summary_path, summary)

    seed_bank = generate_seed_bank(
        target_count=VALIDATION_COUNT,
        ensemble_size=32,
        master_seed=KL_MASTER_SEED,
    )
    seed_path = _write_npy_exclusive(output / "static_seed_bank.npy", seed_bank)
    seed_sha256 = sha256_path(seed_path)
    compressed = {
        "schema_version": 1,
        "scope": "closed_static_Gaussian_KL_compressed_forecast_85604",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "validation_truth_read": False,
        "target_frames": [498, 624],
        "target_frame_index": list(VALIDATION_TARGETS),
        "target_count": VALIDATION_COUNT,
        "ensemble_size": 32,
        "field_order": list(B5_FIELDS),
        "sample_shape": list(SAMPLE_SHAPE),
        "rank": int(selector["rank"]),
        "rank_selection": selector,
        "basis": summary["basis"],
        "seed_bank": {
            "path": str(seed_path.resolve()),
            "sha256": seed_sha256,
            "shape": [VALIDATION_COUNT, 32],
            "dtype": "uint64",
            "generator": "numpy_random_PCG64",
            "master_seed": KL_MASTER_SEED,
        },
        "H1_validation_forecast": {
            "path": str(verified["H1_validation_forecast"]),
            "sha256": args.h1_validation_forecast_sha256,
            "target_frames": [498, 624],
            "shape": [VALIDATION_COUNT, *SAMPLE_SHAPE],
        },
        "formula": (
            "H1_mean_plus_axisymmetric_training_bias_plus_sum_j_"
            "sqrt_eigenvalue_j_times_PCG64_standard_normal_coefficient_"
            "times_KL_mode_j"
        ),
        "covariance_empirical_mean_added_to_forecast": False,
        "truth_or_absolute_time_or_diagnostics_enter_generation": False,
        "checkpoint_loaded": False,
        "model_inference_performed": False,
        "training_performed": False,
    }
    compressed_path = output / "compressed_forecast_manifest.json"
    write_strict_json_atomic(compressed_path, compressed)
    closure = {
        "schema_version": 1,
        "scope": "residual_KL_pretruth_closure_85604",
        "closed_utc": _utc_now(),
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "validation_truth_read": False,
        "basis": {
            "path": summary["basis"]["path"],
            "sha256": summary["basis"]["sha256"],
        },
        "training_summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_path(summary_path),
        },
        "seed_bank": {
            "path": str(seed_path.resolve()),
            "sha256": seed_sha256,
        },
        "compressed_forecast_manifest": {
            "path": str(compressed_path.resolve()),
            "sha256": sha256_path(compressed_path),
        },
        "selected_static_rank": int(selector["rank"]),
        "rank_selected_from_training_eigenvalues_only": True,
        "immutable_before_validation_truth_open": True,
    }
    closure_path = output / "pretruth_closure.json"
    write_strict_json_atomic(closure_path, closure)
    result = {
        "schema_version": 1,
        "scope": "residual_KL_pretruth_build_85604",
        "status": "completed_and_closed_before_validation_truth",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "validation_truth_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "wall_seconds": time.monotonic() - wall_started,
        "positive_rank": int(summary["positive_rank"]),
        "selected_static_rank": int(selector["rank"]),
        "pretruth_closure": {
            "path": str(closure_path.resolve()),
            "sha256": sha256_path(closure_path),
        },
        "checkpoint_loaded": False,
        "model_inference_performed": False,
        "optimizer_or_trainable_parameter_created": False,
        "training_performed": False,
        "physics_metric_used_as_training_loss": False,
        "O3_launched": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
    }
    result_path = output / "pretruth_result.json"
    write_strict_json_atomic(result_path, result)
    indexed = [
        output / "training_kl_basis.h5",
        summary_path,
        seed_path,
        compressed_path,
        closure_path,
        result_path,
    ]
    artifact_index = output / "artifact_sha256.txt"
    with artifact_index.open("x", encoding="utf-8") as handle:
        for path in sorted(indexed, key=lambda value: value.name):
            handle.write(f"{sha256_path(path)}  {path.resolve()}\n")
    _progress(
        "completed_pretruth_closure",
        result_sha256=sha256_path(result_path),
        closure_sha256=sha256_path(closure_path),
        basis_sha256=summary["basis"]["sha256"],
        positive_rank=int(summary["positive_rank"]),
        selected_static_rank=int(selector["rank"]),
    )


if __name__ == "__main__":
    main()
