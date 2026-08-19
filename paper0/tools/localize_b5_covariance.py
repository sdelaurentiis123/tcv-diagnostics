#!/usr/bin/env python3
"""Localize the frozen B5 M32 one-step covariance failure on 85604 only."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "paper0-b5-covariance-localization-v1"


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_field_metrics import b2_region_masks  # noqa: E402
from tcv_diagnostics.b5_covariance_localization import (  # noqa: E402
    B5_COVARIANCE_FIELDS,
    B5_COVARIANCE_TOROIDAL_BANDS,
    CovarianceSummaryAccumulator,
    MarginalAnchorAccumulator,
    TransportCovarianceAccumulator,
    association_summary,
    axisymmetric_bias,
    blockwise_l3_summary,
    classify_localization,
    dependence_distance_summary,
    deterministic_field_error_summary,
    deterministic_toroidal_summary,
    exact_separatrix_local_contributions,
    field_variogram_score,
    gauge_fix_fields,
    subtract_axisymmetric_bias,
    training_frozen_ar1_coefficients,
    training_frozen_ar1_prediction,
)
from tcv_diagnostics.b5_residual_edm_forecast import B5ForecastArtifact  # noqa: E402
from tcv_diagnostics.b5_residual_edm_full_training import (  # noqa: E402
    B5_FULL_VALIDATION_TARGETS,
)
from tcv_diagnostics.b5_residual_forecast import (  # noqa: E402
    B5TrainingForecastArtifact,
)
from tcv_diagnostics.b5_residual_audit import (  # noqa: E402
    cross_field_statistics as legacy_cross_field_statistics,
    spatial_autocorrelation as legacy_spatial_autocorrelation,
)
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


EXPECTED_MANIFEST_SHA256 = (
    "ca60f75de26e6f6af087d1bb5a0af6ead516c6be9445ea226e6281d18c92a7a0"
)
EXPECTED_PROTOCOL_SHA256 = (
    "fa319b8608e3dfc5248f3cfb05e070531409f05be76ca0347c6136a7765f7e8d"
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
OBJECT_ORDER = (
    "training_H1_residual",
    "validation_H1_residual",
    "B5_ensemble_anomaly",
    "B5_innovation",
)
OBJECT_LABELS = {
    "training_H1_residual": "training H1 residual",
    "validation_H1_residual": "validation H1 residual",
    "B5_ensemble_anomaly": "B5 ensemble anomaly",
    "B5_innovation": "B5 innovation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization-manifest", type=Path, required=True)
    parser.add_argument("--localization-manifest-sha256", required=True)
    parser.add_argument("--localization-protocol", type=Path, required=True)
    parser.add_argument("--localization-protocol-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--b5-forecast", type=Path, required=True)
    parser.add_argument("--b5-forecast-sha256", required=True)
    parser.add_argument("--b5-seed-bank", type=Path, required=True)
    parser.add_argument("--b5-seed-bank-sha256", required=True)
    parser.add_argument("--h1-forecast", type=Path, required=True)
    parser.add_argument("--h1-forecast-sha256", required=True)
    parser.add_argument("--b5-score", type=Path, required=True)
    parser.add_argument("--b5-score-sha256", required=True)
    parser.add_argument("--b5-gate", type=Path, required=True)
    parser.add_argument("--b5-gate-sha256", required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--training-audit-sha256", required=True)
    parser.add_argument("--training-raw", type=Path, required=True)
    parser.add_argument("--training-raw-sha256", required=True)
    parser.add_argument("--h1-training-forecast", type=Path, required=True)
    parser.add_argument("--h1-training-forecast-sha256", required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
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
    if tuple(B5_FULL_VALIDATION_TARGETS) != TARGETS:
        raise RuntimeError("B5 full validation targets differ from localization")
    if (
        args.localization_manifest_sha256 != EXPECTED_MANIFEST_SHA256
        or args.localization_protocol_sha256 != EXPECTED_PROTOCOL_SHA256
        or manifest.get("protocol_status")
        != "preexecution_amendment_adds_existing_training_H1_forecast_for_gauge_consistent_drift_reference"
        or manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("B5 covariance-localization authority differs")
    if tuple(manifest.get("data", {}).get("fields", ())) != B5_COVARIANCE_FIELDS:
        raise RuntimeError("B5 covariance-localization field order differs")
    data = manifest["data"]
    if (
        data.get("validation_targets") != [498, 624]
        or data.get("validation_target_count") != 126
        or data.get("chronological_blocks") != [list(item) for item in BLOCK_INTERVALS]
        or data.get("history_probe_targets") != [499, 624]
        or data.get("volume_shape") != [5, 64, 32, 88]
        or data.get("native_toroidal_cells") != 81
        or data.get("ensemble_size") != 32
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
        or data.get("absolute_time_input_used") is not False
        or data.get("guard_frames_read_allowed") is not False
    ):
        raise RuntimeError("B5 covariance-localization data contract differs")
    forbidden = set(manifest.get("forbidden_scope", ()))
    required_forbidden = {
        "checkpoint_loading",
        "model_inference",
        "forecast_mutation",
        "model_training",
        "O3_fixed_block_forecast",
        "assimilation",
        "diagnostic_ranking",
        "85606_access",
    }
    if not required_forbidden.issubset(forbidden):
        raise RuntimeError("B5 covariance-localization closed scope differs")
    locks = manifest["evidence_locks"]
    checks = {
        "B5_forecast": (args.b5_forecast, args.b5_forecast_sha256),
        "B5_scientific_sampler_seed_bank": (
            args.b5_seed_bank,
            args.b5_seed_bank_sha256,
        ),
        "H1_validation_forecast": (args.h1_forecast, args.h1_forecast_sha256),
        "B5_score": (args.b5_score, args.b5_score_sha256),
        "B5_final_gate": (args.b5_gate, args.b5_gate_sha256),
        "training_residual_audit": (
            args.training_audit,
            args.training_audit_sha256,
        ),
        "training_residual_sufficient_statistics": (
            args.training_raw,
            args.training_raw_sha256,
        ),
        "H1_training_forecast": (
            args.h1_training_forecast,
            args.h1_training_forecast_sha256,
        ),
        "native_truth_result": (
            args.native_truth_result,
            args.native_truth_result_sha256,
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
            raise RuntimeError(f"B5 evidence lock digest differs for {name}")
        if _resolve_locked_path(str(record["path"])).resolve() != Path(path).resolve():
            raise RuntimeError(f"B5 evidence lock path differs for {name}")
    dataset = locks["model_dataset"]
    if (
        Path(dataset["root"]).resolve() != Path(args.artifact_root).resolve()
        or dataset.get("manifest_sha256")
        != "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18"
        or dataset.get("normalization_sha256")
        != "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7"
    ):
        raise RuntimeError("B5 model-dataset lock differs")
    execution = manifest["execution"]
    if (
        execution.get("os") != "Rocky_Linux_9"
        or execution.get("accelerator") != "none_CPU_only"
        or execution.get("wandb_online_required") is not True
    ):
        raise RuntimeError("B5 localization execution contract differs")


def require_rocky9_cpu_only() -> None:
    release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if (
        release.get("ID") != "rocky"
        or release.get("VERSION_ID", "").split(".")[0] != "9"
    ):
        raise RuntimeError("B5 covariance localization requires Rocky Linux 9")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in ("", "NoDevFiles"):
        raise RuntimeError("B5 covariance localization is CPU-only")


def _new_bundle(
    region_masks_xy: Mapping[str, np.ndarray]
) -> CovarianceSummaryAccumulator:
    return CovarianceSummaryAccumulator(
        region_masks_xy=region_masks_xy,
        volume_shape=VOLUME_SHAPE,
    )


def _update_in_chunks(
    accumulator: CovarianceSummaryAccumulator,
    values: np.ndarray,
    *,
    chunk_size: int = 4,
) -> None:
    for start in range(0, values.shape[0], int(chunk_size)):
        accumulator.update(values[start : start + int(chunk_size)])


def _finalize_named_bundle(
    name: str,
    accumulator: CovarianceSummaryAccumulator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record, raw = accumulator.finalize()
    return record, {f"{name}__{key}": value for key, value in raw.items()}


def _block_name(start: int, stop: int) -> str:
    return f"frames_{int(start)}_{int(stop) - 1}"


def _block_index(target: int) -> int:
    matches = [
        index
        for index, (start, stop) in enumerate(BLOCK_INTERVALS)
        if start <= int(target) < stop
    ]
    if len(matches) != 1:
        raise ValueError(f"target {target} does not belong to one frozen block")
    return matches[0]


def _load_training_raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _load_standardized_truth(catalog: Any) -> np.ndarray:
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
    if values.shape != (126, 5, *VOLUME_SHAPE):
        raise RuntimeError("validation truth tensor shape differs")
    return np.ascontiguousarray(values, dtype=np.float32)


def _decode_members(standardized: np.ndarray, normalization: Any) -> np.ndarray:
    values = np.asarray(standardized, dtype=np.float32)
    if values.ndim != 5 or values.shape[1:] != (5, *VOLUME_SHAPE):
        raise ValueError("standardized member tensor shape differs")
    decoded = np.stack(
        [
            normalization.records[field].decode(values[:, channel])
            for channel, field in enumerate(B5_COVARIANCE_FIELDS)
        ],
        axis=1,
    )
    return np.asarray(decoded, dtype=np.float64)


def _evaluate_transport_fields(
    physical_model88: np.ndarray,
    geometry: Any,
) -> tuple[dict[str, np.ndarray], float]:
    physical = np.asarray(physical_model88, dtype=np.float64)
    if physical.ndim != 5 or physical.shape[1:] != (5, *VOLUME_SHAPE):
        raise ValueError("physical transport tensor must be [sample,5,64,32,88]")
    native = periodic_resample_float32(physical[:, :4], 81, axis=-1).astype(np.float64)
    state = direct_pressure_transport_state(
        native[:, 0], native[:, 1], native[:, 2], native[:, 3]
    )
    evaluated = evaluate_transport_state(state, geometry)
    return exact_separatrix_local_contributions(
        evaluated,
        strict_face_mask=geometry.strict_face_mask,
        separatrix_face_mask=geometry.separatrix_face_mask,
    )


def _evaluate_native_truth_transport(
    native_truth: Mapping[str, np.ndarray],
    geometry: Any,
) -> tuple[dict[str, np.ndarray], float]:
    evaluated = evaluate_transport_state(
        direct_pressure_transport_state(
            native_truth["Ne"],
            native_truth["Pe"],
            native_truth["Pi"],
            native_truth["phi"],
        ),
        geometry,
    )
    return exact_separatrix_local_contributions(
        evaluated,
        strict_face_mask=geometry.strict_face_mask,
        separatrix_face_mask=geometry.separatrix_face_mask,
    )


def _integrated(local: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        quantity: np.sum(values, axis=(1, 2), dtype=np.float64)
        for quantity, values in local.items()
    }


def _deterministic_transport_metrics(
    prediction: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if (
        tuple(prediction) != TRANSPORT_QUANTITIES
        or tuple(truth) != TRANSPORT_QUANTITIES
    ):
        raise ValueError("deterministic transport quantity order differs")
    result: dict[str, Any] = {}
    for quantity in TRANSPORT_QUANTITIES:
        candidate = np.asarray(prediction[quantity], dtype=np.float64)
        observed = np.asarray(truth[quantity], dtype=np.float64)
        if candidate.shape != observed.shape or candidate.ndim != 1:
            raise ValueError("deterministic integrated transport shapes differ")
        error = candidate - observed
        denominator = float(np.sum(observed * observed))
        result[quantity] = {
            "target_count": int(candidate.size),
            "RMSE": float(np.sqrt(np.mean(error * error))),
            "MAE": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "relative_L2": (
                float(np.sqrt(np.sum(error * error) / denominator))
                if denominator > 0.0
                else None
            ),
        }
    return result


def _safe_relative_improvement(reference: float, candidate: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(candidate) or reference <= 0.0:
        raise ValueError("relative-improvement inputs are invalid")
    return float((reference - candidate) / reference)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(str(name))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_figure(fig: Any, output: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        metadata = (
            {"Software": "tcv-diagnostics Paper 0"}
            if suffix == "png"
            else {"Date": None, "Creator": "tcv-diagnostics Paper 0"}
        )
        fig.savefig(
            path,
            dpi=190 if suffix == "png" else None,
            bbox_inches="tight",
            metadata=metadata,
        )
        paths.append(path.name)
    plt.close(fig)
    return paths


def _plot_spatial_acf(
    output: Path,
    covariance_objects: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    fig, axes = plt.subplots(3, 5, figsize=(19, 10), sharey=True)
    colors = ("#6b7280", "#111827", "#2563eb", "#dc2626")
    for row, axis_name in enumerate(("x", "y", "stored_toroidal_z")):
        for column, field in enumerate(B5_COVARIANCE_FIELDS):
            ax = axes[row, column]
            for object_name, color in zip(OBJECT_ORDER, colors):
                record = covariance_objects[object_name]["spatial_autocorrelation"][
                    axis_name
                ]
                ax.plot(
                    record["lags_cells"],
                    record["fields"][field]["correlation"],
                    label=OBJECT_LABELS[object_name],
                    color=color,
                    linewidth=1.7,
                )
            ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
            ax.axhline(0.1, color="#d1d5db", linewidth=0.7, linestyle="--")
            ax.axhline(-0.1, color="#d1d5db", linewidth=0.7, linestyle="--")
            ax.set_title(f"{field}: {axis_name}")
            if column == 0:
                ax.set_ylabel("pooled normalized correlation")
            if row == 2:
                ax.set_xlabel("lag (stored grid cells)")
            ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        "B5 covariance localization: spatial correlation of residual objects\n"
        "phi is gauge-fixed; realized residuals/innovations are axisymmetric-bias centered",
        y=1.02,
    )
    fig.tight_layout()
    return _save_figure(fig, output, "b5-covariance-spatial-acf")


def _plot_cross_field(
    output: Path,
    covariance_objects: Mapping[str, Mapping[str, Any]],
    distances: Mapping[str, Any],
) -> list[str]:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    image = None
    for index, object_name in enumerate(OBJECT_ORDER):
        matrix = np.asarray(
            covariance_objects[object_name]["cross_field"]["global"][
                "correlation_matrix"
            ]
        )
        image = axes[index].imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        for row in range(5):
            for column in range(5):
                axes[index].text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        axes[index].set_xticks(range(5), B5_COVARIANCE_FIELDS, rotation=45)
        axes[index].set_yticks(range(5), B5_COVARIANCE_FIELDS)
        axes[index].set_title(OBJECT_LABELS[object_name])
    regions = list(distances["cross_field"])
    drift = [
        distances["cross_field"][name]["training_to_validation_H1_residual_RMS"]
        for name in regions
    ]
    anomaly = [
        distances["cross_field"][name]["B5_anomaly_to_validation_H1_residual_RMS"]
        for name in regions
    ]
    positions = np.arange(len(regions))
    axes[4].barh(
        positions + 0.18, drift, height=0.35, label="training→validation drift"
    )
    axes[4].barh(positions - 0.18, anomaly, height=0.35, label="B5 anomaly→validation")
    axes[4].set_yticks(
        positions, [name.replace("_", " ") for name in regions], fontsize=7
    )
    axes[4].set_xlabel("off-diagonal RMS distance")
    axes[4].set_title("Regional mismatch reference")
    axes[4].legend(fontsize=7, frameon=False)
    axes[4].grid(axis="x", alpha=0.2)
    if image is not None:
        fig.colorbar(image, ax=axes[:4], shrink=0.72, label="pooled field correlation")
    fig.suptitle(
        "Cross-field dependence: global matrices and geometry-resolved mismatch",
        y=1.03,
    )
    return _save_figure(fig, output, "b5-covariance-cross-field")


def _plot_toroidal_power(
    output: Path,
    covariance_objects: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    objects = (
        "validation_H1_residual",
        "B5_ensemble_anomaly",
        "B5_innovation",
    )
    bands = tuple(B5_COVARIANCE_TOROIDAL_BANDS)
    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    width = 0.24
    positions = np.arange(len(bands))
    for field_index, field in enumerate(B5_COVARIANCE_FIELDS):
        ax = axes[field_index]
        for object_index, object_name in enumerate(objects):
            record = covariance_objects[object_name]["toroidal_support"]["fields"][
                field
            ]["bands"]
            values = [record[band]["power_fraction"] for band in bands]
            ax.bar(
                positions + (object_index - 1) * width,
                values,
                width=width,
                label=OBJECT_LABELS[object_name],
            )
        ax.set_ylabel(f"{field}\npower fraction")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(
        positions,
        [
            "k=0\nn=0",
            "k=1–3\nn=5–15",
            "k=4–5\nn=20–25",
            "k=6–7\nn=30–35",
            "k≥8\nn≥40",
        ],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        "Parseval-weighted toroidal support (stored k maps to full-torus n=5k)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return _save_figure(fig, output, "b5-covariance-toroidal-power")


def _plot_transport_covariance(
    output: Path,
    transport: Mapping[str, Any],
) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, quantity in zip(axes.reshape(-1), TRANSPORT_QUANTITIES):
        record = transport["quantities"][quantity]["covariance_decomposition"]
        values = [
            record["local_corrected_spread_skill_ratio"],
            record["integrated_corrected_spread_skill_ratio"],
            record["counterfactual_local_spread_skill_after_same_factor"],
        ]
        bars = ax.bar(
            ["local SSR", "integrated SSR", "local SSR after\nintegrated scalar match"],
            values,
            color=("#2563eb", "#dc2626", "#9333ea"),
        )
        ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
        ax.axhspan(0.8, 1.25, color="#16a34a", alpha=0.08)
        ax.set_title(quantity.replace("_", " "))
        ax.set_ylabel("corrected spread / RMSE")
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )
        ax.text(
            0.02,
            0.96,
            "Kens/Kerr = "
            f"{record['ensemble_to_error_coherence_multiplier_ratio']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    fig.suptitle(
        "Exact confined-separatrix transport covariance decomposition\n"
        "Each nonlinear transport operator is evaluated member by member",
        y=1.02,
    )
    fig.tight_layout()
    return _save_figure(fig, output, "b5-covariance-separatrix-transport")


def _plot_variogram(
    output: Path,
    variogram: Mapping[str, Any],
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    regions = list(variogram["field"]["aggregate_region_mean"])
    values = [variogram["field"]["aggregate_region_mean"][name] for name in regions]
    axes[0].bar(np.arange(len(regions)), values, color="#2563eb")
    axes[0].set_xticks(
        np.arange(len(regions)),
        [name.replace("_", " ") for name in regions],
        rotation=50,
        ha="right",
    )
    axes[0].set_ylabel("order-1 field variogram score")
    axes[0].set_title("Five fields, all ten same-cell pairs")
    axes[0].grid(axis="y", alpha=0.2)
    quantities = list(variogram["transport"]["aggregate_equal_lag_mean"])
    transport_values = [
        variogram["transport"]["aggregate_equal_lag_mean"][name] for name in quantities
    ]
    axes[1].bar(np.arange(len(quantities)), transport_values, color="#dc2626")
    axes[1].set_xticks(
        np.arange(len(quantities)),
        [name.replace("_", " ") for name in quantities],
        rotation=30,
        ha="right",
    )
    axes[1].set_ylabel("order-1 transport variogram score")
    axes[1].set_title("Exact-separatrix local contributions, frozen z lags")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Dependence-sensitive variogram scores (diagnostic only; no pass threshold)",
        y=1.03,
    )
    fig.tight_layout()
    return _save_figure(fig, output, "b5-covariance-variogram-scores")


def _plot_history(output: Path, history: Mapping[str, Any]) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    blocks = list(history["chronological_blocks"])
    labels = [name.replace("frames_", "").replace("_", "–") for name in blocks]
    positions = np.arange(len(blocks))
    width = 0.25
    for index, model in enumerate(("H1", "AR1", "B5_field_mean")):
        values = [
            history["chronological_blocks"][block][model]["equal_field_mean_RMSE"]
            for block in blocks
        ]
        axes[0].bar(
            positions + (index - 1) * width,
            values,
            width=width,
            label=model.replace("_", " "),
        )
    axes[0].set_xticks(positions, labels, rotation=35, ha="right")
    axes[0].set_ylabel("equal-field standardized RMSE")
    axes[0].set_title("Causal teacher-forced residual-history probe")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    quantities = list(TRANSPORT_QUANTITIES)
    for index, model in enumerate(("H1", "AR1", "B5_field_mean")):
        values = [
            history["aggregate"][model]["integrated_transport"][quantity]["relative_L2"]
            for quantity in quantities
        ]
        axes[1].bar(
            np.arange(4) + (index - 1) * width,
            values,
            width=width,
            label=model.replace("_", " "),
        )
    axes[1].set_xticks(
        np.arange(4),
        [name.replace("_", " ") for name in quantities],
        rotation=30,
        ha="right",
    )
    axes[1].set_ylabel("integrated transport relative L2")
    axes[1].set_title("Authoritative transport on identical 125 targets")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "What one previous realized H1 residual adds (not an autonomous rollout)",
        y=1.03,
    )
    fig.tight_layout()
    return _save_figure(fig, output, "b5-covariance-history-probe")


def write_figures(output: Path, localization: Mapping[str, Any]) -> list[str]:
    figures: list[str] = []
    figures.extend(_plot_spatial_acf(output, localization["covariance_objects"]))
    figures.extend(
        _plot_cross_field(
            output,
            localization["covariance_objects"],
            localization["dependence_distances"],
        )
    )
    figures.extend(_plot_toroidal_power(output, localization["covariance_objects"]))
    figures.extend(
        _plot_transport_covariance(output, localization["transport_covariance"])
    )
    figures.extend(_plot_variogram(output, localization["variogram_scores"]))
    figures.extend(_plot_history(output, localization["history_probe"]))
    return figures


def _legacy_training_cross_check(
    recomputed: Mapping[str, Any],
    stored: Mapping[str, Any],
    recomputed_bias: np.ndarray,
    stored_bias: np.ndarray,
) -> dict[str, Any]:
    spatial_maximum = 0.0
    spatial_worst = "none"
    for axis in ("x", "y", "stored_toroidal_z"):
        for field in B5_COVARIANCE_FIELDS:
            actual = np.asarray(
                recomputed["spatial_autocorrelation"][axis]["fields"][field][
                    "correlation"
                ]
            )
            expected = np.asarray(
                stored["spatial_autocorrelation"][axis]["fields"][field]["correlation"]
            )
            difference = float(np.max(np.abs(actual - expected)))
            if difference > spatial_maximum:
                spatial_maximum = difference
                spatial_worst = f"{axis}/{field}"
    cross_maximum = 0.0
    cross_worst = "none"
    for region in stored["cross_field"]:
        actual = np.asarray(recomputed["cross_field"][region]["correlation_matrix"])
        expected = np.asarray(stored["cross_field"][region]["correlation_matrix"])
        difference = float(np.max(np.abs(actual - expected)))
        if difference > cross_maximum:
            cross_maximum = difference
            cross_worst = str(region)
    bias_difference = np.abs(recomputed_bias - stored_bias)
    bias_maximum = float(np.max(bias_difference))
    bias_index = tuple(
        int(value)
        for value in np.unravel_index(
            int(np.argmax(bias_difference)), bias_difference.shape
        )
    )
    passed = spatial_maximum <= 2e-6 and cross_maximum <= 2e-6 and bias_maximum <= 2e-6
    if not passed:
        raise RuntimeError(
            "reconstructed ungauged training residual fails legacy audit: "
            f"spatial_max={spatial_maximum:.17g} at {spatial_worst}; "
            f"cross_field_max={cross_maximum:.17g} at {cross_worst}; "
            f"axisymmetric_bias_max={bias_maximum:.17g} at {bias_index}; "
            "frozen_tolerance=1.9999999999999999e-06"
        )
    return {
        "passed": True,
        "spatial_correlation_maximum_absolute_difference": spatial_maximum,
        "cross_field_correlation_maximum_absolute_difference": cross_maximum,
        "axisymmetric_bias_maximum_absolute_difference": bias_maximum,
        "tolerance": 2e-6,
        "legacy_statistics_used_as_phi_gauge_fixed_reference": False,
        "verification_estimator": "exact_legacy_full_tensor_direct_dot_product",
    }


def _legacy_training_batch_record(
    fluctuation: np.ndarray,
    region_masks_xy: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Recompute the integrity anchor with its exact historical implementation.

    The frozen audit evaluated the complete 430-target tensor in one call.
    Chunked FFT and Gram accumulation follows the same mathematical formula,
    but it does not reproduce that historical floating-point reduction closely
    enough on the real tensor to serve as a provenance anchor.
    """

    spatial: dict[str, Any] = {}
    for axis in ("x", "y", "stored_toroidal_z"):
        spatial[axis], _ = legacy_spatial_autocorrelation(fluctuation, axis=axis)
    cross_field, _ = legacy_cross_field_statistics(
        fluctuation,
        region_masks_xy=region_masks_xy,
    )
    return {
        "spatial_autocorrelation": spatial,
        "cross_field": cross_field,
    }


def _toroidal_power_ratios(
    covariance_objects: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reference = covariance_objects["validation_H1_residual"]["toroidal_support"]
    result: dict[str, Any] = {}
    for object_name in ("B5_ensemble_anomaly", "B5_innovation"):
        candidate = covariance_objects[object_name]["toroidal_support"]
        fields: dict[str, Any] = {}
        for field in B5_COVARIANCE_FIELDS:
            bands = {}
            for band in B5_COVARIANCE_TOROIDAL_BANDS:
                numerator = candidate["fields"][field]["bands"][band][
                    "mean_parseval_power_density"
                ]
                denominator = reference["fields"][field]["bands"][band][
                    "mean_parseval_power_density"
                ]
                bands[band] = {
                    "candidate_to_validation_H1_residual_absolute_power_ratio": (
                        float(numerator / denominator) if denominator > 0.0 else None
                    )
                }
            fields[field] = {"bands": bands}
        result[object_name] = {"fields": fields}
    return result


def main() -> None:
    args = parse_args()
    wall_started = time.monotonic()
    verify_checkout(args.paper0_commit)
    require_rocky9_cpu_only()
    manifest_path = verify_input(
        args.localization_manifest,
        args.localization_manifest_sha256,
        "localization manifest",
    )
    protocol_path = verify_input(
        args.localization_protocol,
        args.localization_protocol_sha256,
        "localization protocol",
    )
    manifest = load_strict_json(manifest_path)
    validate_authority(manifest, args)
    verified_paths = {
        "localization_manifest": manifest_path,
        "localization_protocol": protocol_path,
        "B5_score": verify_input(args.b5_score, args.b5_score_sha256, "B5 score"),
        "B5_gate": verify_input(args.b5_gate, args.b5_gate_sha256, "B5 gate"),
        "training_audit": verify_input(
            args.training_audit, args.training_audit_sha256, "training audit"
        ),
        "training_raw": verify_input(
            args.training_raw, args.training_raw_sha256, "training raw"
        ),
        "native_truth_result": verify_input(
            args.native_truth_result,
            args.native_truth_result_sha256,
            "native truth result",
        ),
        "geometry_manifest": verify_input(
            args.geometry_manifest,
            args.geometry_manifest_sha256,
            "geometry manifest",
        ),
        "geometry": verify_input(args.geometry, args.geometry_sha256, "geometry"),
    }
    b5_score = load_strict_json(verified_paths["B5_score"])
    b5_gate = load_strict_json(verified_paths["B5_gate"])
    training_audit = load_strict_json(verified_paths["training_audit"])
    training_raw = _load_training_raw(verified_paths["training_raw"])
    if (
        b5_gate.get("post_gate_instruction")
        != "B5_one_step_gate_failed_localize_without_retuning"
        or b5_gate.get("status") != "completed_failed_frozen_B5_one_seed_gate"
        or b5_gate.get("O3_launch_allowed") is not False
        or b5_gate.get("additional_seed_training_authorized") is not False
        or b5_gate.get("held_out_85606_read") is not False
        or training_audit.get("target_frames") != [2, 432]
        or training_audit.get("validation_frames_read") is not False
        or training_audit.get("held_out_85606_read") is not False
    ):
        raise RuntimeError("B5 stop gate or training residual authority differs")
    output = Path(args.output_directory)
    assert_development_path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _progress("verified_authority", output=str(output))

    catalog = load_official_catalog(args.artifact_root)
    geometry_manifest = load_strict_json(verified_paths["geometry_manifest"])
    geometry = load_transport_geometry(
        geometry_path=verified_paths["geometry"],
        geometry_manifest=geometry_manifest,
    )
    flattened_masks = b2_region_masks(geometry.region_masks, n_z=88)
    region_masks_xy: dict[str, np.ndarray] = {}
    for name, mask in flattened_masks.items():
        volume = np.asarray(mask, dtype=bool).reshape(*VOLUME_SHAPE)
        if not np.all(volume == volume[..., :1]):
            raise RuntimeError(f"geometry region {name} is not toroidally invariant")
        region_masks_xy[name] = np.asarray(volume[..., 0], dtype=bool)
    if tuple(("global", *region_masks_xy)) != tuple(
        manifest["cross_field_covariance"]["regions"]
    ):
        raise RuntimeError("authoritative region order differs")

    validation_truth = _load_standardized_truth(catalog)
    with O2ForecastArtifact(
        args.h1_forecast,
        expected_sha256=args.h1_forecast_sha256,
        target_frames=TARGETS,
    ) as artifact:
        validation_h1 = artifact.read(0, len(TARGETS))
    if validation_h1.shape != validation_truth.shape:
        raise RuntimeError("validation H1/truth shapes differ")

    training_truth_dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="train",
        frames=range(2, 432),
        augment=False,
        seed=1701,
    )
    try:
        training_truth = np.stack(
            [
                training_truth_dataset[index]["volume"]
                for index in range(len(training_truth_dataset))
            ]
        )
    finally:
        training_truth_dataset.close()
    with B5TrainingForecastArtifact(
        args.h1_training_forecast,
        expected_sha256=args.h1_training_forecast_sha256,
        target_frames=range(2, 432),
    ) as artifact:
        training_h1 = np.empty_like(training_truth)
        for start in range(0, 430, 8):
            training_h1[start : start + 8] = artifact.read(start, min(start + 8, 430))
    if training_h1.shape != (430, 5, *VOLUME_SHAPE):
        raise RuntimeError("training H1 forecast shape differs")
    _progress("loaded_truth_and_deterministic_forecasts")

    training_raw_residual = np.asarray(training_truth - training_h1, dtype=np.float32)
    stored_training_bias = np.asarray(
        training_raw["axisymmetric_residual_bias__field_x_y"], dtype=np.float64
    )
    recomputed_training_bias = axisymmetric_bias(training_raw_residual)
    training_legacy_fluctuation = subtract_axisymmetric_bias(
        training_raw_residual, recomputed_training_bias
    )
    legacy_record = _legacy_training_batch_record(
        training_legacy_fluctuation,
        region_masks_xy,
    )
    legacy_cross_check = _legacy_training_cross_check(
        legacy_record,
        training_audit,
        recomputed_training_bias,
        stored_training_bias,
    )
    del training_legacy_fluctuation

    training_gauge_residual = np.asarray(
        gauge_fix_fields(training_truth) - gauge_fix_fields(training_h1),
        dtype=np.float32,
    )
    training_gauge_bias = axisymmetric_bias(training_gauge_residual)
    training_gauge_fluctuation = subtract_axisymmetric_bias(
        training_gauge_residual, training_gauge_bias
    )
    training_bundle = _new_bundle(region_masks_xy)
    _update_in_chunks(training_bundle, training_gauge_fluctuation)
    training_record, training_bundle_raw = _finalize_named_bundle(
        "training_H1_residual", training_bundle
    )
    del training_truth, training_h1, training_gauge_fluctuation
    _progress("reconstructed_gauge_consistent_training_reference")

    validation_truth_gauge = gauge_fix_fields(validation_truth)
    validation_h1_gauge = gauge_fix_fields(validation_h1)
    validation_h1_residual_raw = np.asarray(
        validation_truth - validation_h1, dtype=np.float32
    )
    validation_h1_residual = np.asarray(
        validation_truth_gauge - validation_h1_gauge, dtype=np.float32
    )
    validation_h1_bias = axisymmetric_bias(validation_h1_residual)
    validation_h1_fluctuation = subtract_axisymmetric_bias(
        validation_h1_residual, validation_h1_bias
    )
    validation_block_bundles = [_new_bundle(region_masks_xy) for _ in BLOCK_INTERVALS]
    for index, (start, stop) in enumerate(BLOCK_INTERVALS):
        _update_in_chunks(
            validation_block_bundles[index],
            validation_h1_fluctuation[start - 498 : stop - 498],
        )
    validation_bundle = _new_bundle(region_masks_xy)
    for bundle in validation_block_bundles:
        validation_bundle.merge(bundle)
    validation_record, validation_bundle_raw = _finalize_named_bundle(
        "validation_H1_residual", validation_bundle
    )
    validation_block_records: dict[str, Any] = {}
    all_raw: dict[str, np.ndarray] = {
        **training_bundle_raw,
        **validation_bundle_raw,
        "training_gauge_axisymmetric_bias__field_x_y": training_gauge_bias,
        "validation_H1_axisymmetric_bias__field_x_y": validation_h1_bias,
        "legacy_training_axisymmetric_bias__field_x_y": stored_training_bias,
    }
    for (start, stop), bundle in zip(BLOCK_INTERVALS, validation_block_bundles):
        name = _block_name(start, stop)
        record, raw = _finalize_named_bundle(f"validation_H1_residual__{name}", bundle)
        validation_block_records[name] = record
        all_raw.update(raw)
    del validation_h1_fluctuation

    native_truth_catalog = NativeTruthCatalog(
        load_strict_json(verified_paths["native_truth_result"])
    )
    native_truth_all = native_truth_catalog.read(498, 624)
    b5_mean = np.empty_like(validation_truth)
    innovation = np.empty_like(validation_truth)
    anomaly_bundle = _new_bundle(region_masks_xy)
    anomaly_block_bundles = [_new_bundle(region_masks_xy) for _ in BLOCK_INTERVALS]
    marginal_anchor = MarginalAnchorAccumulator(
        region_mask_xy=region_masks_xy["eligible_union"]
    )
    transport_accumulator = TransportCovarianceAccumulator(
        quantities=TRANSPORT_QUANTITIES,
        rows=16,
        n_z=81,
    )
    field_variogram_rows: list[dict[str, Any]] = []
    field_association_variance: list[float] = []
    field_association_error: list[float] = []
    transport_association_variance = {name: [] for name in TRANSPORT_QUANTITIES}
    transport_association_error = {name: [] for name in TRANSPORT_QUANTITIES}
    truth_integrated = {name: [] for name in TRANSPORT_QUANTITIES}
    b5_memberwise_mean_integrated = {name: [] for name in TRANSPORT_QUANTITIES}
    b5_field_mean_integrated = {name: [] for name in TRANSPORT_QUANTITIES}
    maximum_anomaly_mean = 0.0
    maximum_transport_closure = 0.0

    with B5ForecastArtifact(
        args.b5_forecast,
        expected_sha256=args.b5_forecast_sha256,
        target_frames=TARGETS,
        seed_bank_path=args.b5_seed_bank,
        seed_bank_sha256=args.b5_seed_bank_sha256,
    ) as b5_artifact:
        for target_index, target in enumerate(TARGETS):
            stored = b5_artifact.read(target_index, target_index + 1)
            members = np.asarray(stored[0, :, 0], dtype=np.float32)
            members_gauge = gauge_fix_fields(members)
            mean_gauge = np.mean(members_gauge, axis=0, dtype=np.float64).astype(
                np.float32
            )
            anomalies = np.asarray(members_gauge - mean_gauge[None], dtype=np.float32)
            maximum_anomaly_mean = max(
                maximum_anomaly_mean,
                float(np.max(np.abs(np.mean(anomalies, axis=0, dtype=np.float64)))),
            )
            b5_mean[target_index] = np.mean(members, axis=0, dtype=np.float64)
            innovation[target_index] = validation_truth_gauge[target_index] - mean_gauge
            marginal_anchor.update(members_gauge, validation_truth_gauge[target_index])
            anomaly_block_bundles[_block_index(target)].update(anomalies)
            field_scores = field_variogram_score(
                members_gauge,
                validation_truth_gauge[target_index],
                region_masks_xy=region_masks_xy,
            )
            field_variogram_rows.append({"target_frame": target, **field_scores})
            field_association_variance.append(
                float(np.mean(np.var(members_gauge, axis=0, ddof=1)))
            )
            field_association_error.append(
                float(
                    np.mean(
                        np.square(mean_gauge - validation_truth_gauge[target_index])
                    )
                )
            )

            physical_members = _decode_members(members, catalog.normalization)
            local_forecast, closure = _evaluate_transport_fields(
                physical_members, geometry
            )
            native_one = {
                name: values[target_index : target_index + 1]
                for name, values in native_truth_all.items()
            }
            local_truth_with_time, truth_closure = _evaluate_native_truth_transport(
                native_one, geometry
            )
            local_truth = {
                name: values[0] for name, values in local_truth_with_time.items()
            }
            maximum_transport_closure = max(
                maximum_transport_closure, closure, truth_closure
            )
            transport_accumulator.update(
                target_frame=target,
                forecast=local_forecast,
                truth=local_truth,
            )
            member_integrated = _integrated(local_forecast)
            target_integrated = {
                name: float(np.sum(values, dtype=np.float64))
                for name, values in local_truth.items()
            }
            physical_field_mean = np.mean(
                physical_members, axis=0, keepdims=True, dtype=np.float64
            )
            local_field_mean, mean_closure = _evaluate_transport_fields(
                physical_field_mean, geometry
            )
            maximum_transport_closure = max(maximum_transport_closure, mean_closure)
            field_mean_integrated = _integrated(local_field_mean)
            for quantity in TRANSPORT_QUANTITIES:
                ensemble_values = member_integrated[quantity]
                target_value = target_integrated[quantity]
                ensemble_mean_value = float(np.mean(ensemble_values))
                transport_association_variance[quantity].append(
                    float(np.var(ensemble_values, ddof=1))
                )
                transport_association_error[quantity].append(
                    float((ensemble_mean_value - target_value) ** 2)
                )
                truth_integrated[quantity].append(target_value)
                b5_memberwise_mean_integrated[quantity].append(ensemble_mean_value)
                b5_field_mean_integrated[quantity].append(
                    float(field_mean_integrated[quantity][0])
                )
            if target_index == 0 or (target_index + 1) % 10 == 0:
                _progress(
                    "streaming_B5_covariance",
                    completed_targets=target_index + 1,
                    total_targets=126,
                )

    if maximum_anomaly_mean > 2e-6:
        raise RuntimeError("B5 ensemble anomalies fail numerical mean closure")
    marginal_record = marginal_anchor.finalize()
    expected_anchor = b5_score["field_and_marginal_calibration"]["regions"][
        "eligible_union"
    ]["aggregate"]
    anchor_differences = {
        "equal_channel_ensemble_mean_RMSE": abs(
            marginal_record["equal_channel_ensemble_mean_RMSE"]
            - expected_anchor["equal_channel_ensemble_mean_rmse"]
        ),
        "equal_channel_corrected_spread_skill_ratio": abs(
            marginal_record["equal_channel_corrected_spread_skill_ratio"]
            - expected_anchor["equal_channel_corrected_spread_skill_ratio"]
        ),
    }
    if max(anchor_differences.values()) > 2e-6:
        raise RuntimeError("B5 marginal integrity anchors do not reproduce")

    for bundle in anomaly_block_bundles:
        anomaly_bundle.merge(bundle)
    anomaly_record, anomaly_raw = _finalize_named_bundle(
        "B5_ensemble_anomaly", anomaly_bundle
    )
    all_raw.update(anomaly_raw)
    anomaly_block_records: dict[str, Any] = {}
    for (start, stop), bundle in zip(BLOCK_INTERVALS, anomaly_block_bundles):
        name = _block_name(start, stop)
        record, raw = _finalize_named_bundle(f"B5_ensemble_anomaly__{name}", bundle)
        anomaly_block_records[name] = record
        all_raw.update(raw)

    innovation_bias = axisymmetric_bias(innovation)
    innovation_fluctuation = subtract_axisymmetric_bias(innovation, innovation_bias)
    innovation_block_bundles = [_new_bundle(region_masks_xy) for _ in BLOCK_INTERVALS]
    for index, (start, stop) in enumerate(BLOCK_INTERVALS):
        _update_in_chunks(
            innovation_block_bundles[index],
            innovation_fluctuation[start - 498 : stop - 498],
        )
    innovation_bundle = _new_bundle(region_masks_xy)
    for bundle in innovation_block_bundles:
        innovation_bundle.merge(bundle)
    innovation_record, innovation_raw = _finalize_named_bundle(
        "B5_innovation", innovation_bundle
    )
    all_raw.update(innovation_raw)
    all_raw["B5_innovation_axisymmetric_bias__field_x_y"] = innovation_bias
    innovation_block_records: dict[str, Any] = {}
    for (start, stop), block_bundle in zip(BLOCK_INTERVALS, innovation_block_bundles):
        name = _block_name(start, stop)
        record, raw = _finalize_named_bundle(f"B5_innovation__{name}", block_bundle)
        innovation_block_records[name] = record
        all_raw.update(raw)
    del innovation_fluctuation

    transport_record, transport_raw = transport_accumulator.finalize()
    all_raw.update(
        {f"transport__{name}": values for name, values in transport_raw.items()}
    )
    if maximum_transport_closure > 2e-12:
        raise RuntimeError("exact-separatrix local transport closure differs")
    _progress("completed_B5_stream_and_integrity_anchors")

    covariance_objects = {
        "training_H1_residual": training_record,
        "validation_H1_residual": validation_record,
        "B5_ensemble_anomaly": anomaly_record,
        "B5_innovation": innovation_record,
    }
    dependence_distances = dependence_distance_summary(
        training=training_record,
        validation_h1=validation_record,
        b5_anomaly=anomaly_record,
        b5_innovation=innovation_record,
    )
    l3 = blockwise_l3_summary(
        training=training_record,
        validation_h1_blocks=validation_block_records,
        b5_anomaly_blocks=anomaly_block_records,
    )

    field_region_names = tuple(field_variogram_rows[0])[1:]
    field_variogram_aggregate = {
        region: float(np.mean([row[region] for row in field_variogram_rows]))
        for region in field_region_names
    }
    field_variogram_blocks: dict[str, Any] = {}
    for start, stop in BLOCK_INTERVALS:
        selected = [
            row for row in field_variogram_rows if start <= row["target_frame"] < stop
        ]
        field_variogram_blocks[_block_name(start, stop)] = {
            region: float(np.mean([row[region] for row in selected]))
            for region in field_region_names
        }

    transport_variogram_rows: list[dict[str, Any]] = []
    for target_record in transport_record["per_target"]:
        target = int(target_record["target_frame"])
        for quantity in TRANSPORT_QUANTITIES:
            score = target_record["quantities"][quantity]["transport_variogram_score"]
            transport_variogram_rows.append(
                {
                    "target_frame": target,
                    "quantity": quantity,
                    "equal_lag_mean": score["equal_lag_mean"],
                    **score["by_lag"],
                }
            )
    transport_variogram_aggregate = {
        quantity: transport_record["quantities"][quantity]["transport_variogram_score"][
            "equal_lag_mean"
        ]
        for quantity in TRANSPORT_QUANTITIES
    }
    transport_variogram_blocks: dict[str, Any] = {}
    for start, stop in BLOCK_INTERVALS:
        block = _block_name(start, stop)
        transport_variogram_blocks[block] = {}
        for quantity in TRANSPORT_QUANTITIES:
            selected = [
                row
                for row in transport_variogram_rows
                if row["quantity"] == quantity and start <= row["target_frame"] < stop
            ]
            transport_variogram_blocks[block][quantity] = float(
                np.mean([row["equal_lag_mean"] for row in selected])
            )
    variogram_scores = {
        "field": {
            "definition": "order_one_all_ten_same_cell_field_pairs_equal_weight",
            "aggregate_region_mean": field_variogram_aggregate,
            "chronological_blocks": field_variogram_blocks,
            "pass_threshold": None,
        },
        "transport": {
            "definition": "order_one_exact_separatrix_periodic_z_frozen_lags",
            "aggregate_equal_lag_mean": transport_variogram_aggregate,
            "chronological_blocks": transport_variogram_blocks,
            "pass_threshold": None,
        },
        "used_as_training_loss": False,
    }

    coefficients = training_frozen_ar1_coefficients(training_raw)
    ar1_prediction = training_frozen_ar1_prediction(
        h1_mean=validation_h1[1:],
        previous_h1_residual=validation_h1_residual_raw[:-1],
        coefficients=coefficients,
        axisymmetric_training_bias=stored_training_bias,
    )
    history_truth = validation_truth[1:]
    history_predictions = {
        "H1": validation_h1[1:],
        "AR1": ar1_prediction,
        "B5_field_mean": b5_mean[1:],
    }
    deterministic_integrated = {
        "H1": {name: [] for name in TRANSPORT_QUANTITIES},
        "AR1": {name: [] for name in TRANSPORT_QUANTITIES},
    }
    for model_name in ("H1", "AR1"):
        prediction = history_predictions[model_name]
        for start in range(0, prediction.shape[0], 4):
            physical = _decode_members(
                prediction[start : start + 4], catalog.normalization
            )
            local, closure = _evaluate_transport_fields(physical, geometry)
            maximum_transport_closure = max(maximum_transport_closure, closure)
            integrated = _integrated(local)
            for quantity in TRANSPORT_QUANTITIES:
                deterministic_integrated[model_name][quantity].extend(
                    integrated[quantity].tolist()
                )
    deterministic_integrated["B5_field_mean"] = {
        quantity: b5_field_mean_integrated[quantity][1:]
        for quantity in TRANSPORT_QUANTITIES
    }
    history_truth_integrated = {
        quantity: truth_integrated[quantity][1:] for quantity in TRANSPORT_QUANTITIES
    }

    history_aggregate: dict[str, Any] = {}
    for model_name, prediction in history_predictions.items():
        history_aggregate[model_name] = {
            **deterministic_field_error_summary(prediction, history_truth),
            "toroidal": deterministic_toroidal_summary(prediction, history_truth),
            "integrated_transport": _deterministic_transport_metrics(
                {
                    quantity: np.asarray(
                        deterministic_integrated[model_name][quantity],
                        dtype=np.float64,
                    )
                    for quantity in TRANSPORT_QUANTITIES
                },
                {
                    quantity: np.asarray(
                        history_truth_integrated[quantity], dtype=np.float64
                    )
                    for quantity in TRANSPORT_QUANTITIES
                },
            ),
        }
    history_blocks: dict[str, Any] = {}
    history_block_rows: list[dict[str, Any]] = []
    improved_blocks = 0
    for block_start, block_stop in BLOCK_INTERVALS:
        selected_start = max(block_start, 499)
        selected_stop = block_stop
        start_index = selected_start - 499
        stop_index = selected_stop - 499
        block_name = _block_name(selected_start, selected_stop)
        history_blocks[block_name] = {
            "target_frames": [selected_start, selected_stop],
            "target_count": selected_stop - selected_start,
            "first_partial_block_after_dropping_target_498": block_start == 498,
        }
        truth_block = history_truth[start_index:stop_index]
        for model_name, prediction in history_predictions.items():
            field_metrics = deterministic_field_error_summary(
                prediction[start_index:stop_index], truth_block
            )
            spectral = deterministic_toroidal_summary(
                prediction[start_index:stop_index], truth_block
            )
            transport_metrics = _deterministic_transport_metrics(
                {
                    quantity: np.asarray(
                        deterministic_integrated[model_name][quantity][
                            start_index:stop_index
                        ]
                    )
                    for quantity in TRANSPORT_QUANTITIES
                },
                {
                    quantity: np.asarray(
                        history_truth_integrated[quantity][start_index:stop_index]
                    )
                    for quantity in TRANSPORT_QUANTITIES
                },
            )
            history_blocks[block_name][model_name] = {
                **field_metrics,
                "toroidal": spectral,
                "integrated_transport": transport_metrics,
            }
            history_block_rows.append(
                {
                    "block": block_name,
                    "target_start": selected_start,
                    "target_stop_exclusive": selected_stop,
                    "target_count": selected_stop - selected_start,
                    "model": model_name,
                    "equal_field_mean_RMSE": field_metrics["equal_field_mean_RMSE"],
                    "equal_field_mean_MAE": field_metrics["equal_field_mean_MAE"],
                    **{
                        f"{quantity}_transport_relative_L2": transport_metrics[
                            quantity
                        ]["relative_L2"]
                        for quantity in TRANSPORT_QUANTITIES
                    },
                }
            )
        h1_rmse = history_blocks[block_name]["H1"]["equal_field_mean_RMSE"]
        ar1_rmse = history_blocks[block_name]["AR1"]["equal_field_mean_RMSE"]
        improvement = _safe_relative_improvement(h1_rmse, ar1_rmse)
        history_blocks[block_name]["AR1_vs_H1_RMSE_improvement_fraction"] = improvement
        if improvement > 0.0:
            improved_blocks += 1

    aggregate_improvement = _safe_relative_improvement(
        history_aggregate["H1"]["equal_field_mean_RMSE"],
        history_aggregate["AR1"]["equal_field_mean_RMSE"],
    )
    history_probe = {
        "name": "training_frozen_scalar_fieldwise_residual_AR1",
        "teacher_forced": True,
        "autonomous_rollout": False,
        "coefficient_source": "legacy_training_residual_lag1_sufficient_statistics",
        "axisymmetric_bias_source": "legacy_training_residual_sufficient_statistics",
        "coefficients_by_field": {
            field: float(coefficients[index])
            for index, field in enumerate(B5_COVARIANCE_FIELDS)
        },
        "aggregate": history_aggregate,
        "chronological_blocks": history_blocks,
        "AR1_vs_H1_equal_field_RMSE_improvement_fraction": aggregate_improvement,
        "AR1_improved_chronological_comparison_count": improved_blocks,
        "AR1_beats_B5_field_mean_equal_field_RMSE": (
            history_aggregate["AR1"]["equal_field_mean_RMSE"]
            < history_aggregate["B5_field_mean"]["equal_field_mean_RMSE"]
        ),
        "B5_field_mean_definition": (
            "standardized_member_mean_for_standardized_field_metrics_and_"
            "physical_decoded_member_mean_before_transport_operator_for_transport"
        ),
        "B5_predictive_memberwise_transport_mean": _deterministic_transport_metrics(
            {
                quantity: np.asarray(
                    b5_memberwise_mean_integrated[quantity][1:], dtype=np.float64
                )
                for quantity in TRANSPORT_QUANTITIES
            },
            {
                quantity: np.asarray(
                    history_truth_integrated[quantity], dtype=np.float64
                )
                for quantity in TRANSPORT_QUANTITIES
            },
        ),
    }

    field_association = association_summary(
        field_association_variance, field_association_error
    )
    transport_associations = {
        quantity: association_summary(
            transport_association_variance[quantity],
            transport_association_error[quantity],
        )
        for quantity in TRANSPORT_QUANTITIES
    }
    association = {
        "field_global_equal_scalar_weight": field_association,
        "integrated_transport": transport_associations,
        "interpretation": "flow_dependence_only_not_calibration_proof",
    }

    labels = classify_localization(
        transport_quantities=transport_record["quantities"],
        history_aggregate_improvement_fraction=aggregate_improvement,
        history_improved_block_count=improved_blocks,
    )
    labels["L3_field_dependence_mismatch_beyond_within_run_drift"] = l3
    all_l3_identities = {
        *(
            f"spatial:{axis}:{field}"
            for axis in ("x", "y", "stored_toroidal_z")
            for field in B5_COVARIANCE_FIELDS
        ),
        *(f"cross_field:{region}" for region in validation_record["cross_field"]),
    }
    unresolved_l3_identities = sorted(
        identity
        for identity in all_l3_identities
        if l3["direction_counts"].get(identity, 0) < 5
    )
    l5_reasons = []
    if unresolved_l3_identities:
        l5_reasons.append(
            "some_dependence_identities_do_not_exceed_drift_in_five_blocks"
        )
    if (
        not labels["L1_predominantly_amplitude_limited"]["supported"]
        and not labels["L2_covariance_organization_limited"]["supported"]
    ):
        l5_reasons.append("transport_diagnostics_do_not_support_L1_or_L2")
    labels["L5_unresolved_by_one_realized_trajectory"] = {
        "supported": bool(l5_reasons),
        "reasons": l5_reasons,
        "unresolved_dependence_identities": unresolved_l3_identities,
    }

    localization = {
        "schema_version": 1,
        "scope": "B5_read_only_covariance_localization_85604",
        "status": "completed_without_retraining_or_downstream_opening",
        "development_run": "85604",
        "held_out_85606_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "target_frames": [498, 624],
        "target_count": 126,
        "fields": list(B5_COVARIANCE_FIELDS),
        "ensemble_size": 32,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "covariance_objects": covariance_objects,
        "dependence_distances": dependence_distances,
        "blockwise_L3": l3,
        "toroidal_absolute_power_ratios": _toroidal_power_ratios(covariance_objects),
        "variogram_scores": variogram_scores,
        "transport_covariance": transport_record,
        "history_probe": history_probe,
        "spread_error_association": association,
        "interpretation_labels": labels,
        "integrity_anchors": {
            "legacy_training_reconstruction": legacy_cross_check,
            "B5_marginal_recomputation": {
                "passed": True,
                "recomputed": marginal_record,
                "absolute_differences": anchor_differences,
                "tolerance": 2e-6,
            },
            "maximum_absolute_B5_anomaly_member_mean": maximum_anomaly_mean,
            "B5_anomaly_member_mean_tolerance": 2e-6,
            "maximum_exact_separatrix_relative_sum_closure": maximum_transport_closure,
            "exact_separatrix_relative_sum_tolerance": 2e-12,
            "nonlinear_transport_applied_memberwise_before_reduction": True,
        },
        "scientific_boundaries": {
            "conditional_covariance_identified": False,
            "variogram_used_as_training_loss": False,
            "inflated_forecast_written": False,
            "checkpoint_loaded": False,
            "model_inference_performed": False,
            "model_training_performed": False,
            "forecast_mutated": False,
            "additional_seed_trained": False,
            "O3_launched": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "held_out_85606_read": False,
        },
    }

    association_rows = []
    for index, target in enumerate(TARGETS):
        row: dict[str, Any] = {
            "target_frame": target,
            "field_predicted_mean_member_variance": field_association_variance[index],
            "field_ensemble_mean_squared_error": field_association_error[index],
        }
        for quantity in TRANSPORT_QUANTITIES:
            row[
                f"{quantity}_predicted_integrated_variance"
            ] = transport_association_variance[quantity][index]
            row[f"{quantity}_integrated_squared_error"] = transport_association_error[
                quantity
            ][index]
        association_rows.append(row)

    history_per_target_rows: list[dict[str, Any]] = []
    history_truth_gauge = gauge_fix_fields(history_truth)
    for model_name, prediction in history_predictions.items():
        candidate = gauge_fix_fields(prediction)
        error = np.asarray(candidate - history_truth_gauge, dtype=np.float64)
        per_target_rmse = np.sqrt(np.mean(error * error, axis=(1, 2, 3, 4)))
        per_target_mae = np.mean(np.abs(error), axis=(1, 2, 3, 4))
        for index, target in enumerate(range(499, 624)):
            row = {
                "target_frame": target,
                "model": model_name,
                "equal_field_RMSE": float(per_target_rmse[index]),
                "equal_field_MAE": float(per_target_mae[index]),
            }
            for quantity in TRANSPORT_QUANTITIES:
                predicted = float(deterministic_integrated[model_name][quantity][index])
                observed = float(history_truth_integrated[quantity][index])
                row[f"{quantity}_integrated_prediction"] = predicted
                row[f"{quantity}_integrated_truth"] = observed
                row[f"{quantity}_integrated_error"] = predicted - observed
            history_per_target_rows.append(row)

    distance_rows: list[dict[str, Any]] = []
    for axis, records in dependence_distances["spatial"].items():
        for field, record in records.items():
            distance_rows.append(
                {"kind": "spatial", "identity": f"{axis}:{field}", **record}
            )
    for region, record in dependence_distances["cross_field"].items():
        distance_rows.append({"kind": "cross_field", "identity": region, **record})

    for name, values in training_raw.items():
        if name.startswith("temporal_pattern__") or name == (
            "axisymmetric_residual_bias__field_x_y"
        ):
            all_raw[f"legacy_training_input__{name}"] = values
    all_raw["history_AR1_coefficients__field"] = coefficients
    all_raw["association__field_predicted_variance"] = np.asarray(
        field_association_variance, dtype=np.float64
    )
    all_raw["association__field_squared_error"] = np.asarray(
        field_association_error, dtype=np.float64
    )
    for quantity in TRANSPORT_QUANTITIES:
        all_raw[f"association__{quantity}__predicted_variance"] = np.asarray(
            transport_association_variance[quantity], dtype=np.float64
        )
        all_raw[f"association__{quantity}__squared_error"] = np.asarray(
            transport_association_error[quantity], dtype=np.float64
        )

    localization_path = output / "covariance_localization.json"
    write_strict_json_atomic(localization_path, localization)
    raw_path = output / "raw_accumulators.npz"
    with raw_path.open("xb") as handle:
        np.savez_compressed(handle, **all_raw)
    table_paths = {
        "field_variogram_per_target": output / "field_variogram_per_target.csv",
        "transport_variogram_per_target": output / "transport_variogram_per_target.csv",
        "spread_error_association_per_target": output
        / "spread_error_association_per_target.csv",
        "history_per_target": output / "history_per_target.csv",
        "history_chronological_blocks": output / "history_chronological_blocks.csv",
        "dependence_distances": output / "dependence_distances.csv",
    }
    _write_csv(table_paths["field_variogram_per_target"], field_variogram_rows)
    _write_csv(table_paths["transport_variogram_per_target"], transport_variogram_rows)
    _write_csv(table_paths["spread_error_association_per_target"], association_rows)
    _write_csv(table_paths["history_per_target"], history_per_target_rows)
    _write_csv(table_paths["history_chronological_blocks"], history_block_rows)
    _write_csv(table_paths["dependence_distances"], distance_rows)
    figure_names = write_figures(output, localization)
    _progress("wrote_scientific_outputs", figures=len(figure_names))

    scientific_files = [
        localization_path,
        raw_path,
        *table_paths.values(),
        *(output / name for name in figure_names),
    ]
    result = {
        "schema_version": 1,
        "scope": "B5_read_only_covariance_localization_85604",
        "status": "completed_without_retraining_or_downstream_opening",
        "development_run": "85604",
        "held_out_85606_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "completed_utc": _utc_now(),
        "wall_seconds": time.monotonic() - wall_started,
        "target_frames": [498, 624],
        "target_count": 126,
        "ensemble_size": 32,
        "fields": list(B5_COVARIANCE_FIELDS),
        "mode_mapping": "n=5k",
        "localization": {
            "path": str(localization_path.resolve()),
            "sha256": sha256_path(localization_path),
        },
        "raw_accumulators": {
            "path": str(raw_path.resolve()),
            "sha256": sha256_path(raw_path),
            "array_count": len(all_raw),
        },
        "tables": {
            name: {"path": str(path.resolve()), "sha256": sha256_path(path)}
            for name, path in table_paths.items()
        },
        "figures": [
            {
                "path": str((output / name).resolve()),
                "sha256": sha256_path(output / name),
            }
            for name in figure_names
        ],
        "input_sha256": {
            "localization_manifest": args.localization_manifest_sha256,
            "localization_protocol": args.localization_protocol_sha256,
            "B5_forecast": args.b5_forecast_sha256,
            "B5_seed_bank": args.b5_seed_bank_sha256,
            "H1_validation_forecast": args.h1_forecast_sha256,
            "H1_training_forecast": args.h1_training_forecast_sha256,
            "B5_score": args.b5_score_sha256,
            "B5_gate": args.b5_gate_sha256,
            "training_audit": args.training_audit_sha256,
            "training_raw": args.training_raw_sha256,
            "native_truth_result": args.native_truth_result_sha256,
            "geometry_manifest": args.geometry_manifest_sha256,
            "geometry": args.geometry_sha256,
        },
        "interpretation_labels": labels,
        "integrity_anchors": localization["integrity_anchors"],
        "checkpoint_loaded": False,
        "model_inference_performed": False,
        "model_training_performed": False,
        "forecast_mutated": False,
        "posthoc_calibration_applied": False,
        "additional_seed_trained": False,
        "O3_launched": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "held_out_85606_read": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    indexed_files = [*scientific_files, result_path]
    artifact_index = output / "artifact_sha256.txt"
    with artifact_index.open("x", encoding="utf-8") as handle:
        for path in sorted(indexed_files, key=lambda value: value.name):
            handle.write(f"{sha256_path(path)}  {path.resolve()}\n")
    _progress(
        "completed",
        result_sha256=sha256_path(result_path),
        localization_sha256=sha256_path(localization_path),
        labels={
            name: value.get("supported")
            for name, value in labels.items()
            if isinstance(value, Mapping)
        },
    )


if __name__ == "__main__":
    main()
