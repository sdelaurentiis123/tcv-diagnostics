#!/usr/bin/env python3
"""Run the preregistered 85604-only Paper 0 Phase 3.5 diagnosis."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "svg.hashsalt": "paper0-phase3-5-v1",
    }
)


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b5_covariance_localization import (  # noqa: E402
    exact_separatrix_local_contributions,
)
from tcv_diagnostics.codec_transport import (  # noqa: E402
    TRANSPORT_QUANTITIES,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from tcv_diagnostics.matched_o1_transport import (  # noqa: E402
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import load_strict_json  # noqa: E402
from tcv_diagnostics.model_training_data import load_official_catalog  # noqa: E402
from tcv_diagnostics.resampling import periodic_resample_float32  # noqa: E402
from tcv_diagnostics.phase3_5.context_shuffle import run_b5_context_shuffle  # noqa: E402
from tcv_diagnostics.phase3_5.data import (  # noqa: E402
    ARTIFACT_ROOT,
    GEOMETRY,
    NATIVE_RESULT_SHA256,
    corrected_h1_residuals,
    decode_c5,
    load_c5_frames,
    load_exact_frames,
    load_h1_forecasts,
    verify_primary_artifacts,
)
from tcv_diagnostics.phase3_5.diagnostics import (  # noqa: E402
    CROSS_PAIRS,
    FIELDS,
    MODE_BANDS,
    band_power,
    cross_field_covariance,
    cross_spectrum_summary,
    matrix_relative_distance,
    per_sample_phase_coherence_error,
    raw_scalar_series,
    spectral_band_covariance,
    transport_covariance_summary,
)
from tcv_diagnostics.phase3_5.equivariance import audit_frozen_h1_equivariance  # noqa: E402
from tcv_diagnostics.phase3_5.probes import (  # noqa: E402
    append_target_columns,
    causal_context_features,
    evaluate_chronological_probes_multi,
    nearest_preceding_neighbors,
    neighbor_conditional_variance,
    residual_scalar_targets,
)
from tcv_diagnostics.phase3_5.representations import (  # noqa: E402
    FourierSeparatedRepresentation,
    GlobalPCARepresentation,
    HaarSubbandRepresentation,
    PatchwisePCARepresentation,
    assert_storage_not_above_global,
    centered_variance_capture,
)
from tcv_diagnostics.phase3_5.scope import (  # noqa: E402
    Phase35Block,
    exclusive_output,
    load_phase3_5_protocol,
    sha256_path,
)
from tcv_diagnostics.phase3_5.statistics import (  # noqa: E402
    block_bootstrap_interval,
    effective_sample_record,
    fit_ridge,
    fit_snapshot_subspace_from_raw_gram,
    moving_block_indices,
    permute_complete_blocks,
    predict_ridge,
    regression_metrics,
    raw_sample_gram,
    snapshot_principal_angles,
    snapshot_transfer_capture,
    snapshot_transfer_components,
    standardize_training_features,
)
from tcv_diagnostics.phase3_5.translation import (  # noqa: E402
    circular_toroidal_roll,
    estimate_toroidal_displacement,
    training_field_rms,
)


MANIFEST_RELATIVE = Path("paper0/manifests/phase3_5_cause_localization_85604.json")
NATIVE_RESULT_RELATIVE = Path("paper0/results/phase2_potential_vorticity_all_frame_6893033.json")
GEOMETRY_MANIFEST_RELATIVE = Path("paper0/manifests/phase2_85604_geometry_units.json")
GEOMETRY_MANIFEST_SHA256 = "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
CADENCE_MICROSECONDS = 3.131905426352636
BOOTSTRAP_LENGTHS = (6, 12, 22)
BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_SEED = 2026081935
REPRESENTATION_BUDGETS = (32, 64, 128, 256, 416)
TRANSFER_RANKS = (8, 16, 32, 41)
K4_RANKS = (0, 8, 16, 32, 44, 64, 128, 256, "full_positive_rank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST_RELATIVE)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--scratch-directory", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--wandb-entity", default="sdelaurentiis123-columbia-university")
    parser.add_argument("--wandb-project", default="tcv-diagnostics-paper0")
    parser.add_argument("--wandb-group", default="phase3-5-cause-localization")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def progress(stage: str, **values: Any) -> None:
    print(json.dumps({"utc": utc_now(), "stage": stage, **json_safe(values)}, sort_keys=True), flush=True)


def stable_seed(label: str, *, base: int = BOOTSTRAP_SEED) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int(base) + int.from_bytes(digest[:4], "little")) % (2**32)


def verify_checkout(expected_commit: str) -> dict[str, Any]:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True
    ).stdout
    if actual != expected_commit:
        raise RuntimeError(f"Phase 3.5 commit {actual} differs from {expected_commit}")
    if dirty:
        raise RuntimeError(f"Phase 3.5 checkout is dirty:\n{dirty}")
    release = Path("/etc/os-release").read_text(encoding="utf-8")
    if 'ID="rocky"' not in release and "ID=rocky" not in release:
        raise RuntimeError("Phase 3.5 requires Rocky Linux")
    version = next((line.split("=", 1)[1].strip().strip('"') for line in release.splitlines() if line.startswith("VERSION_ID=")), "")
    if version.split(".")[0] != "9":
        raise RuntimeError("Phase 3.5 requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Phase 3.5 requires exactly one visible Rusty H100")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name:
        raise RuntimeError(f"Phase 3.5 requires H100, found {name}")
    return {"commit": actual, "dirty": False, "rocky_version": version, "GPU": name}


def atomic_json(path: Path, record: Mapping[str, Any]) -> Path:
    target = exclusive_output(path)
    partial = target.with_name(f".{target.name}.partial")
    with partial.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)
    return target


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    target = exclusive_output(path)
    fields = sorted({str(key) for row in rows for key in row})
    partial = target.with_name(f".{target.name}.partial")
    with partial.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)
    return target


def save_figure(fig: plt.Figure, output: Path, stem: str) -> tuple[Path, Path]:
    paths = []
    for extension in ("png", "svg"):
        target = exclusive_output(output / "figures" / f"{stem}.{extension}")
        partial = target.with_name(f".{target.name}.partial")
        with partial.open("xb") as handle:
            fig.savefig(handle, format=extension, bbox_inches="tight")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, target)
        paths.append(target)
    plt.close(fig)
    return paths[0], paths[1]


def atomic_text(path: Path, value: str) -> Path:
    target = exclusive_output(path)
    partial = target.with_name(f".{target.name}.partial")
    with partial.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)
    return target


def atomic_npz(path: Path, **arrays: np.ndarray) -> Path:
    target = exclusive_output(path)
    partial = target.with_name(f".{target.name}.partial")
    with partial.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, target)
    return target


@dataclass
class TransportBundle:
    strict: dict[str, np.ndarray]
    local: dict[str, np.ndarray]
    integrated: dict[str, np.ndarray]
    maximum_closure: float


def transport_from_native(native: Mapping[str, np.ndarray], geometry: Any) -> TransportBundle:
    evaluated = evaluate_transport_state(
        direct_pressure_transport_state(native["Ne"], native["Pe"], native["Pi"], native["phi"]),
        geometry,
    )
    local, closure = exact_separatrix_local_contributions(
        evaluated,
        strict_face_mask=geometry.strict_face_mask,
        separatrix_face_mask=geometry.separatrix_face_mask,
    )
    return TransportBundle(
        strict={name: np.asarray(evaluated[name]["strict_face_contributions"]) for name in TRANSPORT_QUANTITIES},
        local={name: np.asarray(local[name]) for name in TRANSPORT_QUANTITIES},
        integrated={name: np.asarray(evaluated[name]["separatrix_wedge"]) for name in TRANSPORT_QUANTITIES},
        maximum_closure=float(closure),
    )


def transport_from_standardized(standardized: np.ndarray, catalog: Any, geometry: Any) -> TransportBundle:
    physical = decode_c5(catalog, standardized)
    native = periodic_resample_float32(physical[:, :4], 81, axis=-1).astype(np.float64)
    return transport_from_native(
        {"Ne": native[:, 0], "Pe": native[:, 1], "Pi": native[:, 2], "phi": native[:, 3]},
        geometry,
    )


def subtract_transport(first: TransportBundle, second: TransportBundle) -> TransportBundle:
    return TransportBundle(
        strict={name: first.strict[name] - second.strict[name] for name in TRANSPORT_QUANTITIES},
        local={name: first.local[name] - second.local[name] for name in TRANSPORT_QUANTITIES},
        integrated={name: first.integrated[name] - second.integrated[name] for name in TRANSPORT_QUANTITIES},
        maximum_closure=max(first.maximum_closure, second.maximum_closure),
    )


def concatenate_transport(training: TransportBundle, validation: TransportBundle) -> TransportBundle:
    return TransportBundle(
        strict={name: np.concatenate((training.strict[name], validation.strict[name])) for name in TRANSPORT_QUANTITIES},
        local={name: np.concatenate((training.local[name], validation.local[name])) for name in TRANSPORT_QUANTITIES},
        integrated={name: np.concatenate((training.integrated[name], validation.integrated[name])) for name in TRANSPORT_QUANTITIES},
        maximum_closure=max(training.maximum_closure, validation.maximum_closure),
    )


def _bootstrap_mean_row(
    *,
    block: Phase35Block,
    object_name: str,
    metric: str,
    values: np.ndarray,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1 or series.size != len(block.frames):
        raise ValueError("stationarity scalar series differs from block")
    circular = metric.endswith(".phase_radians")
    if circular:
        estimate = float(np.angle(np.mean(np.exp(1j * series))))
        resultant = abs(np.mean(np.exp(1j * series)))
        temporal_sd = float(
            np.sqrt(max(-2.0 * math.log(max(resultant, np.finfo(float).tiny)), 0.0))
        )
    else:
        estimate = float(np.mean(series))
        temporal_sd = float(np.std(series, ddof=1))
    row: dict[str, Any] = {
        "block": block.identifier,
        "split": block.split,
        "start_target": block.start,
        "stop_target_exclusive": block.stop,
        "sample_count": series.size,
        "object": object_name,
        "metric": metric,
        "estimate": estimate,
        "temporal_standard_deviation": temporal_sd,
        "estimator": "circular_mean" if circular else "arithmetic_mean",
    }
    if extra:
        row.update(extra)
    for length in BOOTSTRAP_LENGTHS:
        seed = stable_seed(
            f"stationarity:{block.identifier}:{object_name}:{metric}:L{length}"
        )
        if circular:
            draws = moving_block_indices(
                series.size,
                block_length=min(length, series.size),
                replicates=BOOTSTRAP_REPLICATES,
                seed=seed,
            )
            replicate_angles = np.angle(np.mean(np.exp(1j * series[draws]), axis=1))
            unwrapped = estimate + np.angle(np.exp(1j * (replicate_angles - estimate)))
            row[f"bootstrap_L{length}_lower"] = float(np.quantile(unwrapped, 0.025))
            row[f"bootstrap_L{length}_upper"] = float(np.quantile(unwrapped, 0.975))
        else:
            interval = block_bootstrap_interval(
                series,
                np.mean,
                block_length=min(length, series.size),
                replicates=BOOTSTRAP_REPLICATES,
                seed=seed,
            )
            row[f"bootstrap_L{length}_lower"] = interval["lower"]
            row[f"bootstrap_L{length}_upper"] = interval["upper"]
    return row


def _block_positions(targets: Sequence[int], block: Phase35Block) -> np.ndarray:
    mapping = {int(target): index for index, target in enumerate(targets)}
    try:
        return np.asarray([mapping[target] for target in block.frames], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"block {block.identifier} is absent from target order") from error


def cross_metric_series(values: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    coefficients = np.fft.rfft(array, axis=-1, norm="ortho")
    result: dict[str, np.ndarray] = {}
    for first_name, second_name in CROSS_PAIRS:
        first = FIELDS.index(first_name)
        second = FIELDS.index(second_name)
        a = coefficients[:, first]
        b = coefficients[:, second]
        for band, (lower, upper) in MODE_BANDS.items():
            if lower == 0 or lower >= coefficients.shape[-1]:
                continue
            stop = coefficients.shape[-1] if upper is None else min(upper + 1, coefficients.shape[-1])
            cross = np.sum(a[..., lower:stop] * np.conjugate(b[..., lower:stop]), axis=(1, 2, 3))
            aa = np.sum(np.abs(a[..., lower:stop]) ** 2, axis=(1, 2, 3))
            bb = np.sum(np.abs(b[..., lower:stop]) ** 2, axis=(1, 2, 3))
            prefix = f"{first_name}_{second_name}.{band}"
            result[f"{prefix}.phase_radians"] = np.angle(cross)
            result[f"{prefix}.coherence_squared"] = np.clip(np.abs(cross) ** 2 / (aa * bb), 0.0, 1.0)
    return result


def build_stationarity_rows(
    *,
    protocol: Any,
    targets: Sequence[int],
    physical_truth: np.ndarray,
    residual: np.ndarray,
    exact_state: np.ndarray,
    boundary: np.ndarray,
    truth_transport: TransportBundle,
    error_transport: TransportBundle,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    truth_series, truth_profiles = raw_scalar_series(physical_truth)
    residual_series, residual_profiles = raw_scalar_series(residual)
    truth_series.update(cross_metric_series(physical_truth))
    residual_series.update(cross_metric_series(residual))
    for quantity in TRANSPORT_QUANTITIES:
        truth_series[f"transport.{quantity}.local_RMS"] = np.sqrt(
            np.mean(truth_transport.local[quantity] ** 2, axis=(1, 2))
        )
        truth_series[f"transport.{quantity}.integrated"] = truth_transport.integrated[quantity]
        residual_series[f"transport_error.{quantity}.local_RMS"] = np.sqrt(
            np.mean(error_transport.local[quantity] ** 2, axis=(1, 2))
        )
        residual_series[f"transport_error.{quantity}.integrated"] = error_transport.integrated[quantity]
    exact_names = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort")
    for channel, name in enumerate(exact_names):
        values = exact_state[:, channel]
        truth_series[f"exact_state.{name}.mean"] = np.mean(values, axis=(1, 2, 3))
        truth_series[f"exact_state.{name}.RMS"] = np.sqrt(np.mean(values * values, axis=(1, 2, 3)))
    for side, name in enumerate(("inner", "outer")):
        truth_series[f"Bphi.{name}.mean"] = np.mean(boundary[:, side], axis=1)
        truth_series[f"Bphi.{name}.RMS"] = np.sqrt(np.mean(boundary[:, side] ** 2, axis=1))
        departure = boundary[:, side] - np.mean(boundary[:, side], axis=1, keepdims=True)
        truth_series[f"Bphi.{name}.departure_RMS"] = np.sqrt(np.mean(departure**2, axis=1))

    all_series = {"raw_state": truth_series, "H1_residual": residual_series}
    all_profiles = {"raw_state": truth_profiles, "H1_residual": residual_profiles}
    for block in protocol.blocks:
        selected = _block_positions(targets, block)
        for object_name, series_map in all_series.items():
            for metric, values in series_map.items():
                rows.append(
                    _bootstrap_mean_row(
                        block=block,
                        object_name=object_name,
                        metric=metric,
                        values=np.asarray(values)[selected],
                    )
                )
        for object_name, profile_map in all_profiles.items():
            for metric, values in profile_map.items():
                for radial_index in range(values.shape[1]):
                    rows.append(
                        _bootstrap_mean_row(
                            block=block,
                            object_name=object_name,
                            metric=metric,
                            values=np.asarray(values)[selected, radial_index],
                            extra={"radial_x_index": radial_index},
                        )
                    )
    return rows, all_series


def stationarity_contrasts(
    protocol: Any,
    targets: Sequence[int],
    series_by_object: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    first = next(block for block in protocol.blocks if block.identifier == "T00")
    last = next(block for block in protocol.blocks if block.identifier == "V02")
    first_index = _block_positions(targets, first)
    last_index = _block_positions(targets, last)
    rows = []
    for object_name, series_map in series_by_object.items():
        for metric, complete in series_map.items():
            early = np.asarray(complete, dtype=np.float64)[first_index]
            late = np.asarray(complete, dtype=np.float64)[last_index]
            circular = metric.endswith(".phase_radians")
            if circular:
                early_mean = float(np.angle(np.mean(np.exp(1j * early))))
                late_mean = float(np.angle(np.mean(np.exp(1j * late))))
                estimate = float(np.angle(np.exp(1j * (late_mean - early_mean))))
                early_resultant = abs(np.mean(np.exp(1j * early)))
                late_resultant = abs(np.mean(np.exp(1j * late)))
                early_variance = max(
                    -2.0 * math.log(max(early_resultant, np.finfo(float).tiny)), 0.0
                )
                late_variance = max(
                    -2.0 * math.log(max(late_resultant, np.finfo(float).tiny)), 0.0
                )
                pooled = math.sqrt(0.5 * (early_variance + late_variance))
            else:
                pooled = math.sqrt(0.5 * (float(np.var(early, ddof=1)) + float(np.var(late, ddof=1))))
                estimate = float(np.mean(late) - np.mean(early))
            first_draws = moving_block_indices(
                early.size, block_length=12, replicates=BOOTSTRAP_REPLICATES,
                seed=stable_seed(f"contrast:first:{object_name}:{metric}")
            )
            last_draws = moving_block_indices(
                late.size, block_length=12, replicates=BOOTSTRAP_REPLICATES,
                seed=stable_seed(f"contrast:last:{object_name}:{metric}")
            )
            if circular:
                early_draw = np.angle(np.mean(np.exp(1j * early[first_draws]), axis=1))
                late_draw = np.angle(np.mean(np.exp(1j * late[last_draws]), axis=1))
                wrapped = np.angle(np.exp(1j * (late_draw - early_draw)))
                differences = estimate + np.angle(np.exp(1j * (wrapped - estimate)))
            else:
                differences = np.mean(late[last_draws], axis=1) - np.mean(early[first_draws], axis=1)
            rows.append(
                {
                    "object": object_name,
                    "metric": metric,
                    "first_block": "T00",
                    "last_block": "V02",
                    "difference": estimate,
                    "pooled_temporal_SD": pooled,
                    "standardized_effect": estimate / pooled if pooled > 0.0 else math.nan,
                    "estimator": "circular_difference" if circular else "arithmetic_difference",
                    "bootstrap_L12_lower": float(np.quantile(differences, 0.025)),
                    "bootstrap_L12_upper": float(np.quantile(differences, 0.975)),
                    "excludes_zero": bool(np.quantile(differences, 0.025) > 0 or np.quantile(differences, 0.975) < 0),
                }
            )
    return rows


def build_ess_rows(
    *,
    training_series: Mapping[str, np.ndarray],
    validation_series: Mapping[str, np.ndarray],
    residual_training: np.ndarray,
    residual_raw_gram: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    training_subspace = fit_snapshot_subspace_from_raw_gram(
        residual_raw_gram, np.arange(residual_training.shape[0])
    )
    centered = residual_training.reshape(residual_training.shape[0], -1)
    centered = centered - np.mean(centered, axis=0)
    leading_scores = training_subspace.eigenvectors[:, :8] * np.sqrt(
        (training_subspace.sample_count - 1) * training_subspace.eigenvalues[:8]
    )[None, :]
    combined_training = dict(training_series)
    combined_validation = dict(validation_series)
    for index in range(min(8, leading_scores.shape[1])):
        combined_training[f"H1_residual.leading_KL_projection_{index + 1}"] = leading_scores[:, index]
    n_training = residual_training.shape[0]
    validation_indices = np.arange(n_training, residual_raw_gram.shape[0])
    source_centering = np.eye(n_training) - np.ones((n_training, n_training)) / n_training
    source_block = residual_raw_gram[np.ix_(np.arange(n_training), np.arange(n_training))]
    source_mean_against_centered = np.mean(source_block, axis=0) @ source_centering
    cross = (
        residual_raw_gram[np.ix_(validation_indices, np.arange(n_training))]
        @ source_centering
        - source_mean_against_centered[None, :]
    )
    validation_scores = (
        cross @ training_subspace.eigenvectors[:, :8]
    ) / np.sqrt((n_training - 1) * training_subspace.eigenvalues[:8])[None, :]
    for index in range(min(8, validation_scores.shape[1])):
        combined_validation[f"H1_residual.leading_KL_projection_{index + 1}"] = validation_scores[:, index]
    for region, series_map in (("training", combined_training), ("validation", combined_validation)):
        for name, values in series_map.items():
            series = np.asarray(values, dtype=np.float64)
            if series.ndim != 1 or series.size < 3:
                continue
            for detrend in (False, True):
                record = effective_sample_record(series, detrend=detrend)
                base = {
                    "region": region,
                    "observable": name,
                    "detrended": detrend,
                    "sample_count": record["sample_count"],
                    "method": record["primary_method"],
                    "tau_int": record["primary_tau_int"],
                    "effective_sample_size": record["primary_effective_sample_size"],
                    "last_lag": record["primary_last_lag"],
                    "right_censored": record["primary_right_censored"],
                }
                rows.append(base)
                rows.append(
                    {
                        **base,
                        "method": "self_consistent_window_multiplier_5",
                        "tau_int": record["self_consistent_tau_int"],
                        "effective_sample_size": record["self_consistent_effective_sample_size"],
                        "last_lag": record["self_consistent_last_lag"],
                        "right_censored": record["self_consistent_right_censored"],
                    }
                )
                for window, fixed in record["fixed_windows"].items():
                    rows.append(
                        {
                            **base,
                            "method": f"fixed_window_{window}",
                            "tau_int": fixed["tau_int"],
                            "effective_sample_size": fixed["effective_sample_size"],
                            "last_lag": fixed["last_lag"],
                            "right_censored": False,
                        }
                    )
    return rows


def build_block_transfer_rows(
    *,
    protocol: Any,
    targets: Sequence[int],
    residual: np.ndarray,
    raw_gram: np.ndarray,
    transport_error: TransportBundle,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_map = {int(target): index for index, target in enumerate(targets)}
    matched_indices = {
        block.identifier: np.asarray([target_map[target] for target in block.matched_frames], dtype=np.int64)
        for block in protocol.blocks
    }
    subspaces = {
        block.identifier: fit_snapshot_subspace_from_raw_gram(raw_gram, matched_indices[block.identifier])
        for block in protocol.blocks
    }
    for source_block in protocol.blocks:
        source_name = source_block.identifier
        source_index = matched_indices[source_name]
        source_values = residual[source_index]
        source_mean = np.mean(source_values, axis=0)
        source_field_variance = np.var(source_values, axis=(0, 2, 3, 4), ddof=1)
        source_spectral = spectral_band_covariance(source_values)
        source_cross = cross_field_covariance(source_values)
        source_transport = transport_covariance_summary(
            {name: transport_error.local[name][source_index] for name in TRANSPORT_QUANTITIES},
            {name: transport_error.integrated[name][source_index] for name in TRANSPORT_QUANTITIES},
        )
        for target_block in protocol.blocks:
            target_name = target_block.identifier
            target_index = matched_indices[target_name]
            target_values = residual[target_index]
            for rank in TRANSFER_RANKS:
                captured, total = snapshot_transfer_components(
                    raw_gram, subspaces[source_name], target_index, rank=rank
                )
                capture = float(np.sum(captured) / np.sum(total))
                draws = moving_block_indices(
                    len(target_index), block_length=12, replicates=BOOTSTRAP_REPLICATES,
                    seed=stable_seed(f"transfer:{source_name}:{target_name}:rank{rank}"),
                )
                bootstrap = np.asarray(
                    [np.sum(captured[index]) / np.sum(total[index]) for index in draws], dtype=np.float64
                )
                rows.append(
                    {
                        "source_block": source_name,
                        "target_block": target_name,
                        "source_count": len(source_index),
                        "target_count": len(target_index),
                        "metric_family": "global_PCA_variance_capture",
                        "detail": "source_centered_matched_count",
                        "rank": rank,
                        "value": capture,
                        "bootstrap_L12_lower": float(np.quantile(bootstrap, 0.025)),
                        "bootstrap_L12_upper": float(np.quantile(bootstrap, 0.975)),
                    }
                )
                angle = snapshot_principal_angles(
                    raw_gram, subspaces[source_name], subspaces[target_name], rank=rank
                )
                for detail in ("minimum_cosine", "mean_squared_cosine", "maximum_angle_degrees"):
                    rows.append(
                        {
                            "source_block": source_name,
                            "target_block": target_name,
                            "source_count": len(source_index),
                            "target_count": len(target_index),
                            "metric_family": "principal_angle",
                            "detail": detail,
                            "rank": rank,
                            "value": angle[detail],
                        }
                    )
            target_mean = np.mean(target_values, axis=0)
            target_field_variance = np.var(target_values, axis=(0, 2, 3, 4), ddof=1)
            mean_distance = float(np.linalg.norm((target_mean - source_mean).reshape(-1)))
            mean_scale = float(np.linalg.norm(source_values.reshape(len(source_index), -1))) / math.sqrt(len(source_index))
            rows.append(
                {
                    "source_block": source_name,
                    "target_block": target_name,
                    "source_count": 42,
                    "target_count": 42,
                    "metric_family": "residual_mean_distance",
                    "detail": "L2_over_source_sample_RMS",
                    "value": mean_distance / mean_scale if mean_scale > 0.0 else math.nan,
                }
            )
            for channel, field in enumerate(FIELDS):
                rows.append(
                    {
                        "source_block": source_name,
                        "target_block": target_name,
                        "source_count": 42,
                        "target_count": 42,
                        "metric_family": "field_log_variance_ratio",
                        "detail": field,
                        "value": float(np.log(target_field_variance[channel] / source_field_variance[channel])),
                    }
                )
            target_spectral = spectral_band_covariance(target_values)
            target_cross = cross_field_covariance(target_values)
            target_transport = transport_covariance_summary(
                {name: transport_error.local[name][target_index] for name in TRANSPORT_QUANTITIES},
                {name: transport_error.integrated[name][target_index] for name in TRANSPORT_QUANTITIES},
            )
            for family, source_matrix, target_matrix in (
                ("spectral_band_covariance", source_spectral, target_spectral),
                ("cross_field_covariance", source_cross, target_cross),
                ("local_transport_covariance", source_transport["local_covariance"], target_transport["local_covariance"]),
                ("integrated_transport_covariance", source_transport["integrated_covariance"], target_transport["integrated_covariance"]),
            ):
                rows.append(
                    {
                        "source_block": source_name,
                        "target_block": target_name,
                        "source_count": 42,
                        "target_count": 42,
                        "metric_family": family,
                        "detail": "relative_Frobenius_distance",
                        "value": matrix_relative_distance(source_matrix, target_matrix),
                    }
                )
            for quantity_index, quantity in enumerate(sorted(TRANSPORT_QUANTITIES)):
                first = source_transport["integrated_variance"][quantity_index]
                second = target_transport["integrated_variance"][quantity_index]
                rows.append(
                    {
                        "source_block": source_name,
                        "target_block": target_name,
                        "source_count": 42,
                        "target_count": 42,
                        "metric_family": "integrated_transport_log_variance_ratio",
                        "detail": quantity,
                        "value": float(np.log(second / first)),
                    }
                )

    # Chronological half-block controls use equal 21-sample coefficient budgets.
    for block in protocol.blocks:
        indices = matched_indices[block.identifier]
        halves = ((indices[:21], indices[21:], "first_to_second"), (indices[21:], indices[:21], "second_to_first"))
        for source_index, target_index, direction in halves:
            subspace = fit_snapshot_subspace_from_raw_gram(raw_gram, source_index)
            for rank in (8, 16):
                rows.append(
                    {
                        "source_block": block.identifier,
                        "target_block": block.identifier,
                        "source_count": 21,
                        "target_count": 21,
                        "metric_family": "within_block_half_transfer_control",
                        "detail": direction,
                        "rank": rank,
                        "value": snapshot_transfer_capture(raw_gram, subspace, target_index, rank=rank),
                    }
                )
    return rows


def build_learning_curve_rows(
    *,
    residual_raw_gram: np.ndarray,
    training_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefixes = (42, 84, 126, 168, 210, 252, 294, 336, 378, 420)
    validation_blocks = (
        ("V00", np.arange(training_count, training_count + 42)),
        ("V01", np.arange(training_count + 42, training_count + 84)),
        ("V02", np.arange(training_count + 84, training_count + 126)),
    )
    for prefix in prefixes:
        source = fit_snapshot_subspace_from_raw_gram(residual_raw_gram, np.arange(prefix))
        targets: list[tuple[str, np.ndarray]] = list(validation_blocks)
        if prefix + 42 <= 420:
            targets.insert(0, ("immediately_following_training_block", np.arange(prefix, prefix + 42)))
        for target_name, target_indices in targets:
            for rank in (8, 16, 32):
                rows.append(
                    {
                        "method": "global_PCA_chronological_learning_curve",
                        "source_prefix_targets": prefix,
                        "target_block": target_name,
                        "rank": rank,
                        "value": snapshot_transfer_capture(
                            residual_raw_gram, source, target_indices, rank=rank
                        ),
                    }
                )
    return rows


def run_translation_diagnostics(
    *,
    targets: Sequence[int],
    previous_states: np.ndarray,
    context_states: np.ndarray,
    truth: np.ndarray,
    h1: np.ndarray,
    field_rms: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    if not (
        previous_states.shape == context_states.shape == truth.shape == h1.shape
        and previous_states.shape[0] == len(targets)
    ):
        raise ValueError("translation arrays differ")
    aligned_h1 = np.empty_like(h1, dtype=np.float32)
    transported_persistence = np.empty_like(context_states, dtype=np.float32)
    recent_features = np.empty((len(targets), 4 + len(FIELDS)), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        estimates = {
            "recent_context_motion": estimate_toroidal_displacement(
                previous_states[index], context_states[index], field_rms=field_rms
            ),
            "truth_consecutive_motion": estimate_toroidal_displacement(
                context_states[index], truth[index], field_rms=field_rms
            ),
            "H1_prediction_to_truth_oracle": estimate_toroidal_displacement(
                h1[index], truth[index], field_rms=field_rms
            ),
        }
        for estimator, estimate in estimates.items():
            shared = dict(estimate.shared)
            unambiguous = bool(
                float(shared["peak_correlation"]) >= 0.25
                and float(shared["peak_margin"]) >= 0.05
                and float(shared["normalized_surface_entropy"]) <= 0.90
            )
            rows.append(
                {
                    "target_frame": int(target),
                    "estimator": estimator,
                    "field": "shared_whitened_multichannel",
                    **shared,
                    "unambiguous": unambiguous,
                    "future_truth_used": estimator != "recent_context_motion",
                    "deployable": estimator == "recent_context_motion",
                }
            )
            for channel, field in enumerate(FIELDS):
                rows.append(
                    {
                        "target_frame": int(target),
                        "estimator": estimator,
                        "field": field,
                        **estimate.per_field[channel],
                        "unambiguous": None,
                        "future_truth_used": estimator != "recent_context_motion",
                        "deployable": False,
                    }
                )
        recent = estimates["recent_context_motion"]
        recent_features[index, :4] = (
            float(recent.shared["signed_integer_shift"]),
            float(recent.shared["peak_correlation"]),
            float(recent.shared["peak_margin"]),
            float(recent.shared["normalized_surface_entropy"]),
        )
        recent_features[index, 4:] = [
            float(record["signed_integer_shift"]) for record in recent.per_field
        ]
        truth_shift = int(estimates["truth_consecutive_motion"].shared["signed_integer_shift"])
        oracle_shift = int(estimates["H1_prediction_to_truth_oracle"].shared["signed_integer_shift"])
        aligned_h1[index] = circular_toroidal_roll(h1[index], oracle_shift)
        transported_persistence[index] = circular_toroidal_roll(context_states[index], truth_shift)
        for baseline, candidate in (
            ("H1_unaligned", h1[index]),
            ("H1_truth_assisted_shared_shift", aligned_h1[index]),
            ("persistence_unshifted", context_states[index]),
            ("transported_persistence_truth_assisted_shared_shift", transported_persistence[index]),
        ):
            error = np.asarray(candidate, dtype=np.float64) - np.asarray(truth[index], dtype=np.float64)
            rows.append(
                {
                    "target_frame": int(target),
                    "estimator": "oracle_score",
                    "field": "equal_field",
                    "baseline": baseline,
                    "squared_error": float(np.mean(error * error)),
                    "RMSE": float(np.sqrt(np.mean(error * error))),
                    "future_truth_used": "truth_assisted" in baseline,
                    "deployable": False if "truth_assisted" in baseline else True,
                }
            )
            for channel, field in enumerate(FIELDS):
                rows.append(
                    {
                        "target_frame": int(target),
                        "estimator": "oracle_score",
                        "field": field,
                        "baseline": baseline,
                        "squared_error": float(np.mean(error[channel] ** 2)),
                        "RMSE": float(np.sqrt(np.mean(error[channel] ** 2))),
                        "future_truth_used": "truth_assisted" in baseline,
                        "deployable": False if "truth_assisted" in baseline else True,
                    }
                )
        before = float(np.mean((np.asarray(h1[index], dtype=np.float64) - truth[index]) ** 2))
        after = float(np.mean((np.asarray(aligned_h1[index], dtype=np.float64) - truth[index]) ** 2))
        rows.append(
            {
                "target_frame": int(target),
                "estimator": "alignment_gain",
                "field": "equal_field",
                "baseline": "H1_truth_assisted_shared_shift",
                "alignment_fractional_energy_reduction": (before - after) / before,
                "future_truth_used": True,
                "deployable": False,
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(targets):
            progress("translation_progress", completed=index + 1, total=len(targets))
    return rows, aligned_h1, transported_persistence, recent_features


def append_translation_physics_rows(
    rows: list[dict[str, Any]],
    *,
    targets: Sequence[int],
    truth: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    truth_transport: TransportBundle,
    candidate_transport: Mapping[str, TransportBundle],
) -> None:
    truth_power = band_power(truth)
    truth_cross = cross_spectrum_summary(truth)
    for name, candidate in candidates.items():
        power = band_power(candidate)
        cross = cross_spectrum_summary(candidate)
        for band in MODE_BANDS:
            for channel, field in enumerate(FIELDS):
                rows.append(
                    {
                        "target_frame": "aggregate",
                        "estimator": "physics_score",
                        "baseline": name,
                        "field": field,
                        "metric": f"spectral_power_ratio.{band}",
                        "value": float(np.mean(power[band][:, channel]) / np.mean(truth_power[band][:, channel])),
                        "future_truth_used": "truth_assisted" in name,
                    }
                )
        for pair, band_records in truth_cross.items():
            for band, truth_record in band_records.items():
                candidate_record = cross[pair][band]
                phase_error = abs(float(np.angle(np.exp(1j * (
                    math.radians(candidate_record["phase_degrees"])
                    - math.radians(truth_record["phase_degrees"])
                )))))
                rows.extend(
                    [
                        {
                            "target_frame": "aggregate",
                            "estimator": "physics_score",
                            "baseline": name,
                            "field": pair,
                            "metric": f"cross_phase_absolute_error_radians.{band}",
                            "value": phase_error,
                            "future_truth_used": "truth_assisted" in name,
                        },
                        {
                            "target_frame": "aggregate",
                            "estimator": "physics_score",
                            "baseline": name,
                            "field": pair,
                            "metric": f"coherence_absolute_error.{band}",
                            "value": abs(candidate_record["coherence_squared"] - truth_record["coherence_squared"]),
                            "future_truth_used": "truth_assisted" in name,
                        },
                    ]
                )
        transport = candidate_transport[name]
        for quantity in TRANSPORT_QUANTITIES:
            local_error = transport.local[quantity] - truth_transport.local[quantity]
            integrated_error = transport.integrated[quantity] - truth_transport.integrated[quantity]
            rows.extend(
                [
                    {
                        "target_frame": "aggregate",
                        "estimator": "physics_score",
                        "baseline": name,
                        "field": quantity,
                        "metric": "local_transport_relative_L2",
                        "value": float(np.linalg.norm(local_error) / np.linalg.norm(truth_transport.local[quantity])),
                        "future_truth_used": "truth_assisted" in name,
                    },
                    {
                        "target_frame": "aggregate",
                        "estimator": "physics_score",
                        "baseline": name,
                        "field": quantity,
                        "metric": "integrated_transport_relative_L2",
                        "value": float(np.linalg.norm(integrated_error) / np.linalg.norm(truth_transport.integrated[quantity])),
                        "future_truth_used": "truth_assisted" in name,
                    },
                ]
            )


def run_aligned_k4_ladder(
    *,
    training_residual: np.ndarray,
    validation_residual: np.ndarray,
    aligned_validation_h1: np.ndarray,
    axisymmetric_bias: np.ndarray,
    raw_gram: np.ndarray,
    validation_truth: np.ndarray,
    truth_transport: TransportBundle,
    catalog: Any,
    geometry: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    n_training = training_residual.shape[0]
    if validation_residual.shape[0] != 126 or raw_gram.shape != (n_training + 126,) * 2:
        raise ValueError("aligned K4 arrays differ")
    source = fit_snapshot_subspace_from_raw_gram(raw_gram, np.arange(n_training))
    training_matrix = np.asarray(training_residual, dtype=np.float32).reshape(n_training, -1)
    training_matrix = training_matrix - np.mean(training_matrix, axis=0, keepdims=True)
    u = torch.from_numpy(np.asarray(source.eigenvectors, dtype=np.float32).T).to(device)
    x = torch.from_numpy(training_matrix).to(device)
    denominators = torch.sqrt(
        torch.from_numpy(np.asarray((n_training - 1) * source.eigenvalues, dtype=np.float32)).to(device)
    )[:, None]
    modes = (u @ x) / denominators
    del u, x, denominators
    validation_matrix = torch.from_numpy(
        np.asarray(validation_residual, dtype=np.float32).reshape(126, -1)
    ).to(device)
    coefficients = validation_matrix @ modes.T
    rows: list[dict[str, Any]] = []
    resolved_ranks: list[tuple[str, int]] = []
    for rank in K4_RANKS:
        resolved = source.rank if rank == "full_positive_rank" else min(int(rank), source.rank)
        resolved_ranks.append((str(rank), resolved))
    for label, rank in resolved_ranks:
        if rank == 0:
            projection = np.zeros_like(validation_residual, dtype=np.float32)
        else:
            projected = coefficients[:, :rank] @ modes[:rank]
            projection = projected.reshape(validation_residual.shape).cpu().numpy()
        error = np.asarray(validation_residual, dtype=np.float64) - np.asarray(projection, dtype=np.float64)
        total = float(np.sum(np.asarray(validation_residual, dtype=np.float64) ** 2))
        rows.append(
            {
                "analysis": "aligned_K4_truth_projection_non_deployable",
                "rank_label": label,
                "resolved_rank": rank,
                "metric": "validation_residual_variance_capture",
                "detail": "all_fields",
                "value": 1.0 - float(np.sum(error * error)) / total,
            }
        )
        for channel, field in enumerate(FIELDS):
            field_total = float(np.sum(np.asarray(validation_residual[:, channel], dtype=np.float64) ** 2))
            rows.extend(
                [
                    {
                        "analysis": "aligned_K4_truth_projection_non_deployable",
                        "rank_label": label,
                        "resolved_rank": rank,
                        "metric": "validation_residual_variance_capture",
                        "detail": field,
                        "value": 1.0 - float(np.sum(error[:, channel] ** 2)) / field_total,
                    },
                    {
                        "analysis": "aligned_K4_truth_projection_non_deployable",
                        "rank_label": label,
                        "resolved_rank": rank,
                        "metric": "residual_reconstruction_RMSE",
                        "detail": field,
                        "value": float(np.sqrt(np.mean(error[:, channel] ** 2))),
                    },
                ]
            )
        validation_power = band_power(validation_residual)
        projection_power = band_power(projection)
        for band in MODE_BANDS:
            for channel, field in enumerate(FIELDS):
                denominator = float(np.mean(validation_power[band][:, channel]))
                rows.append(
                    {
                        "analysis": "aligned_K4_truth_projection_non_deployable",
                        "rank_label": label,
                        "resolved_rank": rank,
                        "metric": f"residual_spectral_power_ratio.{band}",
                        "detail": field,
                        "value": float(np.mean(projection_power[band][:, channel]) / denominator) if denominator > 0 else math.nan,
                    }
                )
        if rank > 0:
            truth_cross = cross_spectrum_summary(validation_residual)
            projected_cross = cross_spectrum_summary(projection)
            for pair, bands in truth_cross.items():
                for band, truth_record in bands.items():
                    projected_record = projected_cross[pair][band]
                    rows.append(
                        {
                            "analysis": "aligned_K4_truth_projection_non_deployable",
                            "rank_label": label,
                            "resolved_rank": rank,
                            "metric": f"residual_cross_phase_error_degrees.{band}",
                            "detail": pair,
                            "value": abs(float(np.degrees(np.angle(np.exp(1j * np.radians(
                                projected_record["phase_degrees"] - truth_record["phase_degrees"]
                            )))))),
                        }
                    )
        candidate = np.asarray(
            aligned_validation_h1 + axisymmetric_bias[None, ..., None] + projection,
            dtype=np.float32,
        )
        candidate_transport = transport_from_standardized(candidate, catalog, geometry)
        for quantity in TRANSPORT_QUANTITIES:
            local_error = candidate_transport.local[quantity] - truth_transport.local[quantity]
            integrated_error = candidate_transport.integrated[quantity] - truth_transport.integrated[quantity]
            rows.extend(
                [
                    {
                        "analysis": "aligned_K4_truth_projection_non_deployable",
                        "rank_label": label,
                        "resolved_rank": rank,
                        "metric": "local_transport_relative_L2",
                        "detail": quantity,
                        "value": float(np.linalg.norm(local_error) / np.linalg.norm(truth_transport.local[quantity])),
                    },
                    {
                        "analysis": "aligned_K4_truth_projection_non_deployable",
                        "rank_label": label,
                        "resolved_rank": rank,
                        "metric": "integrated_transport_relative_L2",
                        "detail": quantity,
                        "value": float(np.linalg.norm(integrated_error) / np.linalg.norm(truth_transport.integrated[quantity])),
                    },
                ]
            )
        progress("aligned_K4_rank_complete", rank_label=label, resolved_rank=rank)
    del modes, coefficients, validation_matrix
    torch.cuda.empty_cache()
    return rows


def _representation_block_rows(
    *,
    method: str,
    budget: int,
    block: str,
    source_or_transfer: str,
    reference_residual: np.ndarray,
    reconstruction: np.ndarray,
    source_mean: np.ndarray,
    accounting: Mapping[str, int],
    candidate_transport: TransportBundle | None,
    truth_transport: TransportBundle | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    capture = centered_variance_capture(
        reference_residual, reconstruction, source_mean=source_mean
    )
    common = {
        "method": method,
        "budget_real_coefficients_requested": budget,
        "budget_real_coefficients_actual": accounting["real_coefficients"],
        "learned_basis_float_equivalents": accounting["learned_basis_float_equivalents"],
        "fixed_transform_float_equivalents": accounting["fixed_transform_float_equivalents"],
        "index_integers": accounting["index_integers"],
        "block": block,
        "evaluation": source_or_transfer,
        "target_energy_used_for_allocation": False,
    }
    rows.append({**common, "metric_family": "residual_variance_capture", "detail": "all_fields", "value": capture})
    reference_power = band_power(reference_residual)
    reconstruction_power = band_power(reconstruction)
    for band in MODE_BANDS:
        for channel, field in enumerate(FIELDS):
            denominator = float(np.mean(reference_power[band][:, channel]))
            rows.append(
                {
                    **common,
                    "metric_family": "spectral_band_power_ratio",
                    "detail": f"{field}.{band}",
                    "value": float(np.mean(reconstruction_power[band][:, channel]) / denominator) if denominator > 0 else math.nan,
                }
            )
    rows.append(
        {
            **common,
            "metric_family": "cross_field_covariance_relative_error",
            "detail": "global",
            "value": matrix_relative_distance(
                cross_field_covariance(reference_residual), cross_field_covariance(reconstruction)
            ),
        }
    )
    truth_cross = cross_spectrum_summary(reference_residual)
    reconstructed_cross = cross_spectrum_summary(reconstruction)
    for pair, bands in truth_cross.items():
        for band, truth_record in bands.items():
            candidate_record = reconstructed_cross[pair][band]
            rows.extend(
                [
                    {
                        **common,
                        "metric_family": "density_potential_or_cross_spectrum_phase_error_degrees",
                        "detail": f"{pair}.{band}",
                        "value": abs(float(np.degrees(np.angle(np.exp(1j * np.radians(
                            candidate_record["phase_degrees"] - truth_record["phase_degrees"]
                        )))))),
                    },
                    {
                        **common,
                        "metric_family": "density_potential_or_cross_spectrum_coherence_error",
                        "detail": f"{pair}.{band}",
                        "value": abs(candidate_record["coherence_squared"] - truth_record["coherence_squared"]),
                    },
                ]
            )
    if candidate_transport is not None and truth_transport is not None:
        error = subtract_transport(candidate_transport, truth_transport)
        truth_summary = transport_covariance_summary(truth_transport.local, truth_transport.integrated)
        candidate_summary = transport_covariance_summary(candidate_transport.local, candidate_transport.integrated)
        rows.extend(
            [
                {
                    **common,
                    "metric_family": "local_transport_covariance_relative_error",
                    "detail": "four_quantity_matrix",
                    "value": matrix_relative_distance(
                        truth_summary["local_covariance"], candidate_summary["local_covariance"]
                    ),
                },
                {
                    **common,
                    "metric_family": "integrated_transport_covariance_relative_error",
                    "detail": "four_quantity_matrix",
                    "value": matrix_relative_distance(
                        truth_summary["integrated_covariance"], candidate_summary["integrated_covariance"]
                    ),
                },
            ]
        )
        for quantity in TRANSPORT_QUANTITIES:
            rows.extend(
                [
                    {
                        **common,
                        "metric_family": "local_transport_error_RMS",
                        "detail": quantity,
                        "value": float(np.sqrt(np.mean(error.local[quantity] ** 2))),
                    },
                    {
                        **common,
                        "metric_family": "integrated_transport_error_RMS",
                        "detail": quantity,
                        "value": float(np.sqrt(np.mean(error.integrated[quantity] ** 2))),
                    },
                    {
                        **common,
                        "metric_family": "integrated_transport_variance_ratio",
                        "detail": quantity,
                        "value": float(
                            np.var(candidate_transport.integrated[quantity], ddof=1)
                            / np.var(truth_transport.integrated[quantity], ddof=1)
                        ),
                    },
                ]
            )
    return rows


def run_representation_audit(
    *,
    protocol: Any,
    training_targets: Sequence[int],
    validation_targets: Sequence[int],
    training_residual: np.ndarray,
    validation_residual: np.ndarray,
    aligned_training_residual: np.ndarray,
    aligned_validation_residual: np.ndarray,
    training_h1: np.ndarray,
    validation_h1: np.ndarray,
    aligned_training_h1: np.ndarray,
    aligned_validation_h1: np.ndarray,
    axisymmetric_bias: np.ndarray,
    training_truth_transport: TransportBundle,
    validation_truth_transport: TransportBundle,
    catalog: Any,
    geometry: Any,
) -> list[dict[str, Any]]:
    train_map = {int(target): index for index, target in enumerate(training_targets)}
    source_indices = np.asarray(
        [train_map[target] for block in protocol.training_blocks for target in block.matched_frames],
        dtype=np.int64,
    )
    if source_indices.size != 420:
        raise RuntimeError("H5 matched source is not 420 targets")
    validation_map = {int(target): index for index, target in enumerate(validation_targets)}
    validation_blocks = {
        block.identifier: np.asarray([validation_map[target] for target in block.matched_frames], dtype=np.int64)
        for block in protocol.validation_blocks
    }
    factories: list[tuple[str, Callable[[np.ndarray], Any], bool]] = [
        ("global_PCA_KL", lambda values: GlobalPCARepresentation.fit(values), False),
        ("toroidal_Fourier_separated_complex_KL", FourierSeparatedRepresentation.fit, False),
        ("three_level_Haar_subband_KL", lambda values: HaarSubbandRepresentation.fit(values, levels=3), False),
        (
            "overlapping_xy_patchwise_PCA_full_z",
            lambda values: PatchwisePCARepresentation.fit(
                values, patch_shape=(16, 8, 88), stride=(8, 4, 88)
            ),
            False,
        ),
        ("oracle_shift_aligned_global_PCA", lambda values: GlobalPCARepresentation.fit(values), True),
        (
            "oracle_shift_aligned_Haar_subband_KL",
            lambda values: HaarSubbandRepresentation.fit(values, levels=3),
            True,
        ),
    ]
    rows: list[dict[str, Any]] = []
    bias = axisymmetric_bias[None, ..., None]
    for method, factory, aligned in factories:
        source_residual = (aligned_training_residual if aligned else training_residual)[source_indices]
        source_h1 = (aligned_training_h1 if aligned else training_h1)[source_indices]
        later_residual = aligned_validation_residual if aligned else validation_residual
        later_h1 = aligned_validation_h1 if aligned else validation_h1
        started = time.monotonic()
        representation = factory(source_residual)
        fit_seconds = time.monotonic() - started
        progress("representation_fit_complete", method=method, seconds=fit_seconds)
        for budget in REPRESENTATION_BUDGETS:
            accounting = representation.accounting(budget)
            assert_storage_not_above_global(
                accounting, global_budget=budget, sample_shape=source_residual.shape[1:]
            )
            source_reconstruction = np.asarray(
                representation.reconstruct(source_residual, budget=budget), dtype=np.float32
            )
            source_candidate = np.asarray(source_h1 + bias + source_reconstruction, dtype=np.float32)
            source_transport = transport_from_standardized(source_candidate, catalog, geometry)
            source_truth_transport = TransportBundle(
                strict={name: training_truth_transport.strict[name][source_indices] for name in TRANSPORT_QUANTITIES},
                local={name: training_truth_transport.local[name][source_indices] for name in TRANSPORT_QUANTITIES},
                integrated={name: training_truth_transport.integrated[name][source_indices] for name in TRANSPORT_QUANTITIES},
                maximum_closure=training_truth_transport.maximum_closure,
            )
            rows.extend(
                _representation_block_rows(
                    method=method,
                    budget=budget,
                    block="training_matched_420",
                    source_or_transfer="within_source_reconstruction",
                    reference_residual=source_residual,
                    reconstruction=source_reconstruction,
                    source_mean=representation.mean,
                    accounting=accounting,
                    candidate_transport=source_transport,
                    truth_transport=source_truth_transport,
                )
            )
            validation_reconstruction = np.asarray(
                representation.reconstruct(later_residual, budget=budget), dtype=np.float32
            )
            validation_candidate = np.asarray(later_h1 + bias + validation_reconstruction, dtype=np.float32)
            validation_transport = transport_from_standardized(validation_candidate, catalog, geometry)
            for block, selected in validation_blocks.items():
                candidate_transport = TransportBundle(
                    strict={name: validation_transport.strict[name][selected] for name in TRANSPORT_QUANTITIES},
                    local={name: validation_transport.local[name][selected] for name in TRANSPORT_QUANTITIES},
                    integrated={name: validation_transport.integrated[name][selected] for name in TRANSPORT_QUANTITIES},
                    maximum_closure=validation_transport.maximum_closure,
                )
                truth_transport = TransportBundle(
                    strict={name: validation_truth_transport.strict[name][selected] for name in TRANSPORT_QUANTITIES},
                    local={name: validation_truth_transport.local[name][selected] for name in TRANSPORT_QUANTITIES},
                    integrated={name: validation_truth_transport.integrated[name][selected] for name in TRANSPORT_QUANTITIES},
                    maximum_closure=validation_truth_transport.maximum_closure,
                )
                rows.extend(
                    _representation_block_rows(
                        method=method,
                        budget=budget,
                        block=block,
                        source_or_transfer="chronological_transfer",
                        reference_residual=later_residual[selected],
                        reconstruction=validation_reconstruction[selected],
                        source_mean=representation.mean,
                        accounting=accounting,
                        candidate_transport=candidate_transport,
                        truth_transport=truth_transport,
                    )
                )
            progress("representation_budget_complete", method=method, budget=budget)
        del representation
    return rows


def exact_state_summary_features(exact_state: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    state = np.asarray(exact_state, dtype=np.float64)
    bphi = np.asarray(boundary, dtype=np.float64)
    if state.ndim != 5 or state.shape[1] != 6 or bphi.shape != (state.shape[0], 2, 32):
        raise ValueError("exact-state summary shapes differ")
    columns: list[np.ndarray] = []
    for channel in (3, 5):  # NVe and Vort are new relative to C5P.
        values = state[:, channel]
        columns.append(np.mean(values, axis=(1, 2, 3)))
        columns.append(np.sqrt(np.mean(values * values, axis=(1, 2, 3))))
        profile = np.mean(values, axis=(2, 3)).reshape(state.shape[0], 8, 8).mean(axis=-1)
        columns.extend(profile[:, index] for index in range(8))
        spectrum = np.fft.rfft(values, axis=-1, norm="ortho")
        for lower, upper in ((1, 3), (4, 5), (6, 7), (8, spectrum.shape[-1] - 1)):
            columns.append(np.log(np.maximum(
                np.mean(np.abs(spectrum[..., lower : upper + 1]) ** 2, axis=(1, 2, 3)),
                np.finfo(float).tiny,
            )))
    for side in range(2):
        columns.append(np.mean(bphi[:, side], axis=1))
        columns.append(np.sqrt(np.mean(bphi[:, side] ** 2, axis=1)))
        columns.extend(bphi[:, side, index] for index in range(32))
    return np.column_stack(columns)


def compact_state_embedding(
    c5: np.ndarray,
    *,
    exact_state: np.ndarray | None = None,
    boundary: np.ndarray | None = None,
) -> np.ndarray:
    state = np.asarray(c5, dtype=np.float64)
    if state.ndim != 5 or state.shape[1:] != (5, 64, 32, 88):
        raise ValueError("compact embedding C5 shape differs")

    def transform(values: np.ndarray) -> np.ndarray:
        # Average x and y into exactly 8x8 bins before retaining k=0..7.
        binned = values.reshape(values.shape[0], values.shape[1], 8, 8, 8, 4, 88).mean(axis=(3, 5))
        coefficients = np.fft.rfft(binned, axis=-1, norm="ortho")[..., :8]
        return np.concatenate((coefficients.real, coefficients.imag[..., 1:]), axis=-1).reshape(values.shape[0], -1)

    pieces = [transform(state)]
    if exact_state is not None or boundary is not None:
        exact = np.asarray(exact_state, dtype=np.float64)
        bphi = np.asarray(boundary, dtype=np.float64)
        if exact.shape != (state.shape[0], 6, 64, 32, 88) or bphi.shape != (state.shape[0], 2, 32):
            raise ValueError("exact compact embedding shapes differ")
        pieces.append(transform(exact[:, (3, 5)]))
        pieces.append(bphi.reshape(state.shape[0], -1))
    return np.column_stack(pieces)


def build_probe_targets(
    *,
    residual: np.ndarray,
    truth: np.ndarray,
    h1: np.ndarray,
    transport_error: TransportBundle,
    oracle_shift: np.ndarray,
    alignment_gain: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    base = residual_scalar_targets(residual)
    phase_error, coherence_error = per_sample_phase_coherence_error(truth, h1)
    additions: dict[str, np.ndarray] = {
        "phase_error.Ne_phi": phase_error,
        "coherence_error.Ne_phi": coherence_error,
        "oracle_displacement": oracle_shift,
        "alignment_gain": alignment_gain,
    }
    for quantity in TRANSPORT_QUANTITIES:
        additions[f"local_transport_error_energy.{quantity}"] = np.mean(
            transport_error.local[quantity] ** 2, axis=(1, 2)
        )
        additions[f"integrated_transport_error_squared.{quantity}"] = (
            transport_error.integrated[quantity] ** 2
        )
    return append_target_columns(base, additions)


def aggregate_probe_families(
    targets: np.ndarray,
    names: Sequence[str],
    *,
    reference_targets: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    matrix = np.asarray(targets, dtype=np.float64)
    reference = matrix if reference_targets is None else np.asarray(reference_targets, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != matrix.shape[1]:
        raise ValueError("probe-family reference dimensions differ")
    labels = tuple(str(name) for name in names)
    groups = {
        "residual_field_energy": [index for index, name in enumerate(labels) if name.startswith("residual_energy")],
        "residual_spectral_energy": [index for index, name in enumerate(labels) if name.startswith("residual_spectral_energy")],
        "cross_field_covariance": [index for index, name in enumerate(labels) if name.startswith("residual_cross_covariance")],
        "phase_coherence_error": [index for index, name in enumerate(labels) if name.startswith(("phase_error", "coherence_error"))],
        "local_transport_error": [index for index, name in enumerate(labels) if name.startswith("local_transport")],
        "integrated_transport_error": [index for index, name in enumerate(labels) if name.startswith("integrated_transport")],
        "oracle_displacement": [index for index, name in enumerate(labels) if name == "oracle_displacement"],
        "alignment_gain": [index for index, name in enumerate(labels) if name == "alignment_gain"],
    }
    result = {}
    for family, indices in groups.items():
        if not indices:
            continue
        selected = matrix[:, indices]
        reference_selected = reference[:, indices]
        center = np.mean(reference_selected, axis=0)
        scale = np.std(reference_selected, axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        result[family] = np.mean((selected - center) / scale, axis=1)
    return result


def block_permutation_rows(
    *,
    training_features: np.ndarray,
    validation_features: np.ndarray,
    training_families: Mapping[str, np.ndarray],
    validation_families: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    train, validation = standardize_training_features(training_features, validation_features)
    permutations = permute_complete_blocks(
        (42,) * 10, replicates=200, seed=2026081936
    )
    rows: list[dict[str, Any]] = []
    for family in sorted(training_families):
        y_train = np.asarray(training_families[family], dtype=np.float64)
        y_validation = np.asarray(validation_families[family], dtype=np.float64)
        observed = predict_ridge(train, fit_ridge(train, y_train, alpha=1.0))
        del observed
        null_r2 = np.empty(permutations.shape[0], dtype=np.float64)
        for replicate, permutation in enumerate(permutations):
            coefficient = fit_ridge(train, y_train[permutation], alpha=1.0)
            prediction = predict_ridge(validation, coefficient).reshape(-1)
            null_r2[replicate] = regression_metrics(y_validation, prediction)["R2"]
        rows.append(
            {
                "analysis": "complete_42_sample_block_permutation_null",
                "target_family": family,
                "probe": "context_ridge_alpha_1_fixed_null",
                "replicates": 200,
                "null_R2_median": float(np.median(null_r2)),
                "null_R2_95_upper": float(np.quantile(null_r2, 0.95)),
                "IID_permutation_used": False,
            }
        )
    return rows


def run_context_and_state_probes(
    *,
    protocol: Any,
    training_targets: Sequence[int],
    validation_targets: Sequence[int],
    training_context: np.ndarray,
    validation_context: np.ndarray,
    training_recent_features: np.ndarray,
    validation_recent_features: np.ndarray,
    training_exact_context: np.ndarray,
    validation_exact_context: np.ndarray,
    training_boundary_context: np.ndarray,
    validation_boundary_context: np.ndarray,
    training_target_matrix: np.ndarray,
    validation_target_matrix: np.ndarray,
    target_names: Sequence[str],
    training_all_c5: np.ndarray,
    validation_all_c5: np.ndarray,
    training_all_exact: np.ndarray,
    validation_all_exact: np.ndarray,
    training_all_boundary: np.ndarray,
    validation_all_boundary: np.ndarray,
) -> list[dict[str, Any]]:
    train_map = {int(target): index for index, target in enumerate(training_targets)}
    source_targets = tuple(target for block in protocol.training_blocks for target in block.matched_frames)
    source_index = np.asarray([train_map[target] for target in source_targets], dtype=np.int64)
    if source_index.size != 420:
        raise RuntimeError("probe source is not ten complete 42-target blocks")
    context_train, _ = causal_context_features(
        training_context, displacement_features=training_recent_features
    )
    context_validation, _ = causal_context_features(
        validation_context, displacement_features=validation_recent_features
    )
    exact_train = exact_state_summary_features(training_exact_context, training_boundary_context)
    exact_validation = exact_state_summary_features(validation_exact_context, validation_boundary_context)
    feature_sets = {
        "current_C5P": (context_train[source_index], context_validation),
        "current_C5P_plus_NVe_Vort_Bphi": (
            np.column_stack((context_train[source_index], exact_train[source_index])),
            np.column_stack((context_validation, exact_validation)),
        ),
    }
    rows: list[dict[str, Any]] = []
    validation_blocks = tuple(block.identifier for block in protocol.validation_blocks)
    source_time = np.asarray(source_targets, dtype=np.float64)
    validation_time = np.asarray(validation_targets, dtype=np.float64)
    for feature_set, (train_features, validation_features) in feature_sets.items():
        probe_rows = evaluate_chronological_probes_multi(
            train_features,
            source_time,
            training_target_matrix[source_index],
            validation_features,
            validation_time,
            validation_target_matrix,
            target_names=target_names,
            validation_block_ids=validation_blocks,
        )
        rows.extend(
            {"analysis": "chronological_probe", "feature_set": feature_set, **record}
            for record in probe_rows
        )

    # History comparison uses T01--T09, where every frozen lag is available.
    history_source_targets = tuple(
        target for block in protocol.training_blocks[1:] for target in block.matched_frames
    )
    history_source_index = np.asarray([train_map[target] for target in history_source_targets])
    lags = (1, 2, 4, 8, 16)
    validation_map = {int(target): index for index, target in enumerate(validation_targets)}
    history_validation_targets = tuple(
        target
        for target in validation_targets
        if target - 1 - max(lags) >= 496
    )
    history_validation_index = np.asarray(
        [validation_map[target] for target in history_validation_targets], dtype=np.int64
    )
    history_validation_sizes = tuple(
        sum(block.start <= target < block.stop for target in history_validation_targets)
        for block in protocol.validation_blocks
    )
    if history_validation_sizes != (27, 42, 42):
        raise RuntimeError("legal delayed validation subset differs from frozen boundaries")

    def history_features(
        all_c5: np.ndarray,
        all_exact: np.ndarray,
        all_boundary: np.ndarray,
        targets: Sequence[int],
        frame_origin: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        c5_pieces = []
        exact_pieces = []
        for lag in (0, *lags):
            frames = np.asarray([target - 1 - lag - frame_origin for target in targets], dtype=np.int64)
            if np.any(frames < 0) or np.any(frames >= all_c5.shape[0]):
                raise ValueError("history probe requests unavailable causal frames")
            feature, _ = causal_context_features(all_c5[frames])
            c5_pieces.append(feature)
            exact_pieces.append(exact_state_summary_features(all_exact[frames], all_boundary[frames]))
        return np.column_stack(c5_pieces), np.column_stack((*c5_pieces, *exact_pieces))

    delayed_train, delayed_exact_train = history_features(
        training_all_c5, training_all_exact, training_all_boundary, history_source_targets, 0
    )
    delayed_validation, delayed_exact_validation = history_features(
        validation_all_c5,
        validation_all_exact,
        validation_all_boundary,
        history_validation_targets,
        496,
    )
    current_history_train, _ = causal_context_features(training_context[history_source_index])
    current_history_validation, _ = causal_context_features(
        validation_context[history_validation_index]
    )
    history_sets = {
        "current_C5P_history_matched": (current_history_train, current_history_validation),
        "C5P_delays_1_2_4_8_16": (delayed_train, delayed_validation),
        "exact_summaries_plus_C5P_delays_1_2_4_8_16": (delayed_exact_train, delayed_exact_validation),
    }
    for feature_set, (train_features, validation_features) in history_sets.items():
        probe_rows = evaluate_chronological_probes_multi(
            train_features,
            np.asarray(history_source_targets, dtype=np.float64),
            training_target_matrix[history_source_index],
            validation_features,
            np.asarray(history_validation_targets, dtype=np.float64),
            validation_target_matrix[history_validation_index],
            target_names=target_names,
            validation_block_ids=validation_blocks,
            validation_block_sizes=history_validation_sizes,
            minimum_training_blocks=4,
            block_size=42,
        )
        rows.extend(
            {"analysis": "history_probe", "feature_set": feature_set, **record}
            for record in probe_rows
        )

    # Block-permutation sensitivity uses the primary C5P feature set.
    train_families = aggregate_probe_families(training_target_matrix[source_index], target_names)
    validation_families = aggregate_probe_families(
        validation_target_matrix,
        target_names,
        reference_targets=training_target_matrix[source_index],
    )
    rows.extend(
        block_permutation_rows(
            training_features=context_train[source_index],
            validation_features=context_validation,
            training_families=train_families,
            validation_families=validation_families,
        )
    )

    # Causal nearest-neighbor state-completeness comparison.
    train_c5_embedding = compact_state_embedding(training_context[source_index])
    validation_c5_embedding = compact_state_embedding(validation_context)
    train_exact_embedding = compact_state_embedding(
        training_context[source_index],
        exact_state=training_exact_context[source_index],
        boundary=training_boundary_context[source_index],
    )
    validation_exact_embedding = compact_state_embedding(
        validation_context,
        exact_state=validation_exact_context,
        boundary=validation_boundary_context,
    )
    c5_train, c5_validation = standardize_training_features(
        train_c5_embedding, validation_c5_embedding
    )
    # Exact embeddings have additional columns, so they require their own scale.
    exact_train_scaled, exact_validation_scaled = standardize_training_features(
        train_exact_embedding, validation_exact_embedding
    )
    family_train = aggregate_probe_families(training_target_matrix[source_index], target_names)
    family_validation = aggregate_probe_families(
        validation_target_matrix,
        target_names,
        reference_targets=training_target_matrix[source_index],
    )
    for embedding_name, train_embedding, validation_embedding in (
        ("C5P", c5_train, c5_validation),
        ("C5P_plus_NVe_Vort_Bphi", exact_train_scaled, exact_validation_scaled),
    ):
        for k in (5, 10, 20):
            neighbors, distances = nearest_preceding_neighbors(
                train_embedding,
                np.asarray(source_targets),
                validation_embedding,
                np.asarray(validation_targets),
                k=k,
                minimum_separation=42,
            )
            for family in sorted(family_train):
                for block_index, block_id in enumerate(validation_blocks):
                    selected = slice(block_index * 42, (block_index + 1) * 42)
                    summary = neighbor_conditional_variance(
                        neighbors[selected],
                        family_train[family],
                        family_validation[family][selected],
                    )
                    rows.append(
                        {
                            "analysis": "causal_nearest_neighbor_state_completeness",
                            "feature_set": embedding_name,
                            "target": family,
                            "block": block_id,
                            "neighbor_k": k,
                            "minimum_temporal_exclusion_frames": 42,
                            "mean_neighbor_distance": float(np.nanmean(distances[selected])),
                            **summary,
                        }
                    )
    return rows


def b5_transport_shuffle_callback(catalog: Any, geometry: Any) -> Callable[[np.ndarray, np.ndarray, int], Sequence[Mapping[str, object]]]:
    def callback(correct: np.ndarray, shuffled: np.ndarray, target: int) -> Sequence[Mapping[str, object]]:
        correct_transport = transport_from_standardized(correct, catalog, geometry)
        shuffled_transport = transport_from_standardized(shuffled, catalog, geometry)
        first = transport_covariance_summary(correct_transport.local, correct_transport.integrated)
        second = transport_covariance_summary(shuffled_transport.local, shuffled_transport.integrated)
        rows: list[dict[str, object]] = [
            {
                "family": "local_transport_covariance",
                "quantity": "four_quantity_matrix",
                "correct_context": float(np.linalg.norm(first["local_covariance"])),
                "shuffled_context": float(np.linalg.norm(second["local_covariance"])),
                "relative_change": matrix_relative_distance(first["local_covariance"], second["local_covariance"]),
            },
            {
                "family": "integrated_transport_covariance",
                "quantity": "four_quantity_matrix",
                "correct_context": float(np.linalg.norm(first["integrated_covariance"])),
                "shuffled_context": float(np.linalg.norm(second["integrated_covariance"])),
                "relative_change": matrix_relative_distance(first["integrated_covariance"], second["integrated_covariance"]),
            },
        ]
        for quantity in TRANSPORT_QUANTITIES:
            correct_variance = float(np.var(correct_transport.integrated[quantity], ddof=1))
            shuffled_variance = float(np.var(shuffled_transport.integrated[quantity], ddof=1))
            rows.append(
                {
                    "family": "integrated_transport_variance",
                    "quantity": quantity,
                    "correct_context": correct_variance,
                    "shuffled_context": shuffled_variance,
                    "relative_change": (
                        (shuffled_variance - correct_variance) / correct_variance
                        if correct_variance > 0.0 else math.nan
                    ),
                }
            )
        return rows

    return callback


def make_plots(
    *,
    output: Path,
    stationarity_rows: Sequence[Mapping[str, Any]],
    ess_rows: Sequence[Mapping[str, Any]],
    transfer_rows: Sequence[Mapping[str, Any]],
    translation_rows: Sequence[Mapping[str, Any]],
    equivariance_rows: Sequence[Mapping[str, Any]],
    representation_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    figures: list[dict[str, str]] = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for panel, object_name in zip(axes, ("raw_state", "H1_residual")):
        for field in FIELDS:
            metric = f"{field}.global_fluctuation_RMS"
            selected = [
                row for row in stationarity_rows
                if row.get("object") == object_name and row.get("metric") == metric
            ]
            if selected:
                baseline = float(selected[0]["estimate"])
                centers = [
                    0.5 * (int(row["start_target"]) + int(row["stop_target_exclusive"]) - 1)
                    for row in selected
                ]
                estimates = np.asarray(
                    [float(row["estimate"]) / baseline for row in selected]
                )
                lower = np.asarray(
                    [float(row["bootstrap_L12_lower"]) / baseline for row in selected]
                )
                upper = np.asarray(
                    [float(row["bootstrap_L12_upper"]) / baseline for row in selected]
                )
                panel.errorbar(
                    centers,
                    estimates,
                    yerr=np.vstack(
                        (
                            np.maximum(estimates - lower, 0.0),
                            np.maximum(upper - estimates, 0.0),
                        )
                    ),
                    marker="o",
                    capsize=2,
                    linewidth=1,
                    label=field,
                )
        panel.axvspan(432, 496, color="0.85", label="unread guard" if object_name == "raw_state" else None)
        panel.axhline(1.0, color="0.25", linewidth=0.8)
        panel.set(title=object_name.replace("_", " "), xlabel="85604 target frame", ylabel="fluctuation RMS / T00 value")
        panel.grid(alpha=0.25)
    axes[0].legend(ncol=2)
    paths = save_figure(fig, output, "stationarity-field-rms")
    figures.append({"stem": "stationarity-field-rms", "png": str(paths[0]), "svg": str(paths[1]),
                    "caption": "Chronological block means and non-circular L=12 block-bootstrap intervals for raw-state and H1-residual fluctuation RMS, normalized field-by-field to T00. The shaded guard is not read or interpolated."})

    primary = [
        row for row in ess_rows
        if row.get("region") == "training" and row.get("detrended") is False
        and row.get("method") == "Geyer_initial_positive_pair_sequence"
        and ("residual" in str(row.get("observable")) or "transport" in str(row.get("observable")))
    ]
    primary = sorted(primary, key=lambda row: float(row["effective_sample_size"]))[:24]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(range(len(primary)), [float(row["effective_sample_size"]) for row in primary], color="#4477AA")
    ax.set_yticks(range(len(primary)), [str(row["observable"]) for row in primary], fontsize=7)
    ax.axvline(20, color="#CC3311", linestyle="--", label="ESS=20 evidence threshold")
    ax.invert_yaxis()
    ax.set(xlabel="effective sample size", title="Least effectively sampled training observables")
    ax.legend()
    paths = save_figure(fig, output, "effective-sample-size")
    figures.append({"stem": "effective-sample-size", "png": str(paths[0]), "svg": str(paths[1]),
                    "caption": "Geyer initial-positive-pair ESS for the least sampled residual, mode, and transport series; adjacent frames are not counted as independent shots."})

    block_names = [f"T{index:02d}" for index in range(10)] + [f"V{index:02d}" for index in range(3)]
    matrix = np.full((13, 13), np.nan)
    for row in transfer_rows:
        if row.get("metric_family") == "global_PCA_variance_capture" and int(row.get("rank", -1)) == 41:
            matrix[block_names.index(str(row["source_block"])), block_names.index(str(row["target_block"]))] = float(row["value"])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(0.5, float(np.nanmax(matrix))))
    ax.set_xticks(range(13), block_names, rotation=45)
    ax.set_yticks(range(13), block_names)
    ax.set(xlabel="target block", ylabel="basis source block", title="Rank-41 source-to-target residual variance capture")
    for row_index in range(13):
        for column_index in range(13):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if value < 0.45 * np.nanmax(matrix) else "black",
                )
    fig.colorbar(image, ax=ax, label="captured source-centered target variance")
    paths = save_figure(fig, output, "block-transfer-rank41-heatmap")
    figures.append({"stem": "block-transfer-rank41-heatmap", "png": str(paths[0]), "svg": str(paths[1]),
                    "caption": "Ordered 42-by-42 chronological transfer matrix at equal rank and equal sample count. Rows fit the basis; columns evaluate it."})

    consecutive = [
        row for row in translation_rows
        if row.get("estimator") == "truth_consecutive_motion" and row.get("field") == "shared_whitened_multichannel"
    ]
    score_rows = [
        row for row in translation_rows
        if row.get("estimator") == "oracle_score" and row.get("field") == "equal_field"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].hist([int(row["signed_integer_shift"]) for row in consecutive], bins=np.arange(-44.5, 45.5), color="#228833")
    axes[0].axvspan(7, 14, color="#CCBB44", alpha=0.25, label="preregistered strong range")
    axes[0].axvspan(-14, -7, color="#CCBB44", alpha=0.25)
    axes[0].set(xlabel="shared shift applied to earlier state [z cells]", ylabel="transition count", title="Consecutive-state toroidal displacement")
    axes[0].legend()
    labels = ("H1_unaligned", "H1_truth_assisted_shared_shift", "persistence_unshifted", "transported_persistence_truth_assisted_shared_shift")
    values = [np.mean([float(row["RMSE"]) for row in score_rows if row.get("baseline") == label]) for label in labels]
    axes[1].bar(range(len(labels)), values, color=("#4477AA", "#66CCEE", "#CC6677", "#AA3377"))
    axes[1].set_xticks(range(len(labels)), ("H1", "shifted H1\noracle", "persistence", "transported\npersistence oracle"))
    axes[1].set(ylabel="equal-field standardized RMSE", title="Position-oracle impact")
    paths = save_figure(fig, output, "translation-oracles")
    figures.append({"stem": "translation-oracles", "png": str(paths[0]), "svg": str(paths[1]),
                    "caption": "Shared toroidal motion and truth-assisted position-oracle errors. Oracle bars are diagnostic and nondeployable."})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for panel, scope in zip(axes, ("codec", "H1")):
        selected = [row for row in equivariance_rows if row.get("scope") == scope and row.get("field") == "equal_field"]
        for frame in sorted({int(row["target_frame"]) for row in selected}):
            curve = [float(row["equivariance_relative_error"]) for row in selected if int(row["target_frame"]) == frame]
            panel.plot(range(88), curve, alpha=0.35, linewidth=0.8)
        means = [np.mean([float(row["equivariance_relative_error"]) for row in selected if int(row["shift_cells"]) == shift]) for shift in range(88)]
        panel.plot(range(88), means, color="black", linewidth=2, label="13-state mean")
        panel.set(title=f"{scope} toroidal equivariance", xlabel="integer z shift", ylabel="normalized equivariance error")
        panel.legend()
        panel.grid(alpha=0.2)
    paths = save_figure(fig, output, "equivariance-all-shifts")
    figures.append({"stem": "equivariance-all-shifts", "png": str(paths[0]), "svg": str(paths[1]),
                    "caption": "All 88 periodic toroidal shifts for 13 preregistered states. Every H1 history frame is rolled together."})

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    methods = sorted({str(row["method"]) for row in representation_rows if row.get("evaluation") == "chronological_transfer"})
    for method in methods:
        values = []
        for budget in REPRESENTATION_BUDGETS:
            selected = [
                float(row["value"]) for row in representation_rows
                if row.get("method") == method
                and row.get("metric_family") == "residual_variance_capture"
                and row.get("evaluation") == "chronological_transfer"
                and int(row.get("budget_real_coefficients_requested", -1)) == budget
            ]
            values.append(float(np.mean(selected)))
        ax.plot(REPRESENTATION_BUDGETS, values, marker="o", label=method)
    ax.set(xscale="log", xlabel="real coefficient budget", ylabel="mean V00–V02 source-centered variance capture", title="Matched-budget chronological representation transfer")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    paths = save_figure(fig, output, "representation-transfer")
    figures.append({"stem": "representation-transfer", "png": str(paths[0]), "svg": str(paths[1]),
                    "caption": "Later-block transfer at matched real coefficient budgets. All allocation is based on the 420-target training source only."})

    ridge = [
        row for row in context_rows
        if row.get("analysis") == "chronological_probe" and row.get("probe") == "context_ridge"
    ]
    targets = sorted({str(row["target"]) for row in ridge})
    summary = []
    for target in targets:
        current = [float(row["R2"]) for row in ridge if row.get("target") == target and row.get("feature_set") == "current_C5P"]
        exact = [float(row["R2"]) for row in ridge if row.get("target") == target and row.get("feature_set") == "current_C5P_plus_NVe_Vort_Bphi"]
        if current and exact:
            summary.append((target, np.mean(current), np.mean(exact)))
    summary = sorted(summary, key=lambda item: item[2] - item[1], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    y = np.arange(len(summary))
    ax.scatter([item[1] for item in summary], y, label="C5P", color="#4477AA")
    ax.scatter([item[2] for item in summary], y, label="C5P + NVe/Vort/Bphi", color="#EE6677")
    for index, item in enumerate(summary):
        ax.plot((item[1], item[2]), (index, index), color="0.7", zorder=0)
    ax.set_yticks(y, [item[0] for item in summary], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="mean chronological validation R²", title="Largest exact-state probe changes")
    ax.legend()
    paths = save_figure(fig, output, "context-state-probes")
    figures.append({"stem": "context-state-probes", "png": str(paths[0]), "svg": str(paths[1]),
                    "caption": "Chronological ridge-probe R² with available C5P context and with omitted saved-state summaries added. Negative R² is retained."})
    return figures


TIER_ORDER = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}


def probe_target_family(target: str) -> str:
    """Map scalar diagnostic targets to the preregistered target families."""

    name = str(target)
    prefixes = (
        ("residual_energy.", "residual_field_energy"),
        ("residual_spectral_energy.", "residual_spectral_energy"),
        ("residual_cross_covariance.", "cross_field_covariance"),
        ("phase_error.", "phase_coherence_error"),
        ("coherence_error.", "phase_coherence_error"),
        ("local_transport_error_energy.", "local_transport_error"),
        ("integrated_transport_error_squared.", "integrated_transport_error"),
    )
    for prefix, family in prefixes:
        if name.startswith(prefix):
            return family
    return name


def representation_error_value(row: Mapping[str, Any]) -> float:
    """Put dependence/transport reconstruction metrics on lower-is-better scales."""

    value = float(row["value"])
    if row.get("metric_family") == "integrated_transport_variance_ratio":
        return abs(math.log(value)) if value > 0.0 else math.inf
    return abs(value)


def classify_evidence(
    *,
    contrasts: Sequence[Mapping[str, Any]],
    ess_rows: Sequence[Mapping[str, Any]],
    learning_rows: Sequence[Mapping[str, Any]],
    translation_rows: Sequence[Mapping[str, Any]],
    equivariance_rows: Sequence[Mapping[str, Any]],
    equivariance_curves: Mapping[str, Any],
    representation_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    evidence: list[dict[str, Any]] = []

    material_contrasts = [
        row for row in contrasts
        if row.get("object") == "raw_state"
        and (
            "global_fluctuation_RMS" in str(row.get("metric"))
            or ".integrated" in str(row.get("metric"))
            or str(row.get("metric", "")).startswith("Bphi")
        )
    ]
    shifted = [
        row for row in material_contrasts
        if abs(float(row["standardized_effect"])) > 0.5 and bool(row["excludes_zero"])
    ]
    time_rows = [
        row for row in context_rows
        if row.get("analysis") == "chronological_probe"
        and row.get("feature_set") == "current_C5P"
        and row.get("probe") == "time_only_ridge"
    ]
    time_targets = {
        str(row["target"])
        for row in time_rows
        if float(row.get("R2", -math.inf)) >= 0.10
    }
    nonstationary_strong = len(shifted) >= 3 or len(time_targets) >= 2
    nonstationary_ratio = max(len(shifted) / 3.0, len(time_targets) / 2.0)
    nonstationary_tier = "strong" if nonstationary_strong else "moderate" if nonstationary_ratio >= 0.5 else "weak" if nonstationary_ratio > 0 else "none"
    evidence.append(
        {
            "explanation": "invalid/nonstationary interval",
            "tier": nonstationary_tier,
            "normalized_effect": nonstationary_ratio,
            "summary": f"{len(shifted)} material T00-to-V02 shifts exceed 0.5 pooled SD with block CI excluding zero; {len(time_targets)} targets have time-only R2 >= 0.10.",
            "protocol_invalid": False,
        }
    )

    consecutive = [
        row for row in translation_rows
        if row.get("estimator") == "truth_consecutive_motion" and row.get("field") == "shared_whitened_multichannel"
    ]
    unambiguous_fraction = float(np.mean([bool(row["unambiguous"]) for row in consecutive]))
    median_shift = float(np.median([abs(int(row["signed_integer_shift"])) for row in consecutive]))
    gains = [
        float(row["alignment_fractional_energy_reduction"])
        for row in translation_rows if row.get("estimator") == "alignment_gain"
    ]
    median_gain = float(np.median(gains))
    k4_full = [
        float(row["value"]) for row in translation_rows
        if row.get("analysis") == "aligned_K4_truth_projection_non_deployable"
        and row.get("rank_label") == "full_positive_rank"
        and row.get("metric") == "validation_residual_variance_capture"
        and row.get("detail") == "all_fields"
    ]
    capture_gain = (k4_full[0] - 0.2234) if k4_full else -math.inf
    score_rows = [row for row in translation_rows if row.get("estimator") == "oracle_score" and row.get("field") == "equal_field"]
    persistence = np.mean([float(row["squared_error"]) for row in score_rows if row.get("baseline") == "persistence_unshifted"])
    transported = np.mean([float(row["squared_error"]) for row in score_rows if row.get("baseline") == "transported_persistence_truth_assisted_shared_shift"])
    persistence_gain = float((persistence - transported) / persistence)
    coherent_strong = (
        unambiguous_fraction >= 0.75
        and 7 <= median_shift <= 14
        and median_gain >= 0.25
        and (capture_gain >= 0.15 or persistence_gain >= 0.20)
    )
    coherent_components = np.mean(
        [
            min(unambiguous_fraction / 0.75, 2.0),
            1.0 if 7 <= median_shift <= 14 else max(0.0, 1.0 - abs(median_shift - 10.5) / 10.5),
            max(0.0, median_gain / 0.25),
            max(capture_gain / 0.15, persistence_gain / 0.20, 0.0),
        ]
    )
    coherent_tier = "strong" if coherent_strong else "moderate" if coherent_components >= 0.5 else "weak" if coherent_components > 0 else "none"
    evidence.append(
        {
            "explanation": "coherent transport in an inappropriate Eulerian representation",
            "tier": coherent_tier,
            "normalized_effect": coherent_components,
            "summary": f"unambiguous={unambiguous_fraction:.3f}, median |shift|={median_shift:.2f}/88, median H1 energy reduction={median_gain:.3f}, aligned full-span gain={capture_gain:.3f}, transported-persistence gain={persistence_gain:.3f}.",
        }
    )

    strong_states = 0
    ratio_values = []
    modulo_values = []
    for frame, scopes in equivariance_curves.items():
        triggered = False
        for scope in ("codec", "H1"):
            selected = [
                row for row in equivariance_rows
                if str(row.get("target_frame")) == str(frame)
                and row.get("scope") == scope and row.get("field") == "equal_field"
            ]
            baseline = next(float(row["error_against_rolled_truth"]) for row in selected if int(row["shift_cells"]) == 0)
            median_error = float(np.median([float(row["equivariance_relative_error"]) for row in selected if int(row["shift_cells"]) != 0]))
            ratio = median_error / baseline if baseline > 0 else math.inf
            modulo = scopes[scope]["modulo_4_range_over_median_nonzero"]
            modulo = float(modulo) if modulo is not None else 0.0
            ratio_values.append(ratio)
            modulo_values.append(modulo)
            if ratio >= 0.25 or modulo >= 0.20:
                triggered = True
        strong_states += int(triggered)
    non_equiv_ratio = max(strong_states / 10.0, np.median(ratio_values) / 0.25, np.median(modulo_values) / 0.20)
    non_equiv_tier = "strong" if strong_states >= 10 else "moderate" if non_equiv_ratio >= 0.5 else "weak" if non_equiv_ratio > 0 else "none"
    evidence.append(
        {
            "explanation": "codec or predictor non-equivariance",
            "tier": non_equiv_tier,
            "normalized_effect": non_equiv_ratio,
            "summary": f"{strong_states}/13 representative states cross a frozen equivariance criterion; median equivariance/base-error ratio={np.median(ratio_values):.3f}, median modulo-4 ratio={np.median(modulo_values):.3f}.",
        }
    )

    transfer = [
        row for row in representation_rows
        if row.get("evaluation") == "chronological_transfer"
        and row.get("metric_family") == "residual_variance_capture"
    ]
    global_by_budget = {
        budget: np.mean([float(row["value"]) for row in transfer if row.get("method") == "global_PCA_KL" and int(row["budget_real_coefficients_requested"]) == budget])
        for budget in REPRESENTATION_BUDGETS
    }
    method_gains: dict[str, dict[int, float]] = {}
    for method in {str(row["method"]) for row in transfer if row.get("method") != "global_PCA_KL"}:
        method_gains[method] = {
            budget: np.mean([float(row["value"]) for row in transfer if row.get("method") == method and int(row["budget_real_coefficients_requested"]) == budget]) - global_by_budget[budget]
            for budget in REPRESENTATION_BUDGETS
        }
    best_method = max(method_gains, key=lambda name: max(method_gains[name].values()))
    qualifying_budgets = [
        budget for budget, value in method_gains[best_method].items() if value >= 0.10
    ]
    budgets_above = len(qualifying_budgets)
    best_gain = max(method_gains[best_method].values())
    dependence_groups = {
        "cross_field_covariance": ("cross_field_covariance_relative_error",),
        "cross_spectrum_phase": (
            "density_potential_or_cross_spectrum_phase_error_degrees",
        ),
        "cross_spectrum_coherence": (
            "density_potential_or_cross_spectrum_coherence_error",
        ),
        "local_transport_covariance": ("local_transport_covariance_relative_error",),
        "integrated_transport_covariance": (
            "integrated_transport_covariance_relative_error",
        ),
        "integrated_transport_variance": ("integrated_transport_variance_ratio",),
    }

    def mean_group_error(method: str, budget: int, families: Sequence[str]) -> float:
        selected = [
            representation_error_value(row)
            for row in representation_rows
            if row.get("evaluation") == "chronological_transfer"
            and row.get("method") == method
            and int(row["budget_real_coefficients_requested"]) == budget
            and row.get("metric_family") in families
        ]
        return float(np.mean(selected)) if selected else math.inf

    physics_improvements: dict[str, list[int]] = {}
    for family, metrics in dependence_groups.items():
        improved = [
            budget
            for budget in qualifying_budgets
            if mean_group_error(best_method, budget, metrics)
            < mean_group_error("global_PCA_KL", budget, metrics)
        ]
        if improved:
            physics_improvements[family] = improved
    consistently_improved_families = [
        family
        for family, budgets in physics_improvements.items()
        if len(qualifying_budgets) >= 2
        and all(budget in budgets for budget in qualifying_budgets[:2])
    ]
    representation_ratio = (
        min(max(budgets_above / 2.0, best_gain / 0.10), 2.0) * 0.5
        + min(len(consistently_improved_families) / 2.0, 2.0) * 0.5
    )
    representation_strong = (
        budgets_above >= 2 and len(consistently_improved_families) >= 2
    )
    representation_tier = (
        "strong"
        if representation_strong
        else "moderate"
        if representation_ratio >= 0.5
        else "weak"
        if best_gain > 0 or physics_improvements
        else "none"
    )

    context_probe = [
        row for row in context_rows
        if row.get("analysis") == "chronological_probe"
        and row.get("feature_set") == "current_C5P"
    ]
    conditional_targets = []
    for target in {str(row["target"]) for row in context_probe}:
        by_block: dict[str, dict[str, float]] = {}
        for row in context_probe:
            if str(row.get("target")) == target:
                by_block.setdefault(str(row["block"]), {})[str(row["probe"])] = float(row["R2"])
        if len(by_block) == 3 and all(
            values.get("context_ridge", -math.inf) - values.get("constant", math.inf) >= 0.10
            and values.get("context_ridge", -math.inf) - values.get("time_only_ridge", math.inf) >= 0.05
            for values in by_block.values()
        ):
            conditional_targets.append(target)
    conditional_families = sorted({probe_target_family(target) for target in conditional_targets})
    b5_rows = [row for row in context_rows if row.get("analysis") == "B5_context_shuffle"]
    b5_change = float(np.median([abs(float(row["relative_change"])) for row in b5_rows])) if b5_rows else 0.0
    # B5 is corroborating only and cannot independently promote this mechanism
    # to the strong tier.
    context_ratio = max(len(conditional_families) / 2.0, min(b5_change / 0.10, 1.0))
    context_tier = "strong" if len(conditional_families) >= 2 else "moderate" if context_ratio >= 0.5 else "weak" if context_ratio > 0 else "none"
    evidence.append(
        {
            "explanation": "forecast-state-dependent covariance",
            "tier": context_tier,
            "normalized_effect": context_ratio,
            "summary": f"{len(conditional_targets)} scalar targets spanning {len(conditional_families)} target families satisfy the all-three-block context-probe rule; median fixed-seed B5 covariance-family change={b5_change:.3f} (corroborating only).",
        }
    )

    evidence.append(
        {
            "explanation": "localized or multiscale representation advantage",
            "tier": representation_tier,
            "normalized_effect": representation_ratio,
            "summary": f"best method={best_method}; {budgets_above} budgets improve later variance transfer by >=0.10; maximum gain={best_gain:.3f}; consistently improved dependence/transport families={','.join(consistently_improved_families) if consistently_improved_families else 'none'}.",
        }
    )

    neighbor = [row for row in context_rows if row.get("analysis") == "causal_nearest_neighbor_state_completeness"]
    state_success = []
    for target in {str(row["target"]) for row in neighbor}:
        passed_blocks = 0
        effects = []
        for block in ("V00", "V01", "V02"):
            c5 = [row for row in neighbor if row.get("target") == target and row.get("block") == block and row.get("feature_set") == "C5P" and int(row.get("neighbor_k", -1)) == 10]
            exact = [row for row in neighbor if row.get("target") == target and row.get("block") == block and row.get("feature_set") == "C5P_plus_NVe_Vort_Bphi" and int(row.get("neighbor_k", -1)) == 10]
            if c5 and exact:
                candidate_effects = []
                for metric in ("prediction_RMSE", "conditional_variance"):
                    baseline = float(c5[0][metric])
                    if baseline > 0.0:
                        candidate_effects.append(
                            (baseline - float(exact[0][metric])) / baseline
                        )
                effect = max(candidate_effects, default=-math.inf)
                effects.append(effect)
                passed_blocks += int(effect >= 0.10)
        if passed_blocks == 3:
            state_success.append((target, min(effects)))

    exact_probe_rows = [
        row for row in context_rows
        if row.get("analysis") == "chronological_probe"
        and row.get("probe") == "context_ridge"
    ]
    exact_probe_success = []
    for target in {str(row["target"]) for row in exact_probe_rows}:
        passed = 0
        for block in ("V00", "V01", "V02"):
            current = [
                row for row in exact_probe_rows
                if row.get("target") == target and row.get("block") == block
                and row.get("feature_set") == "current_C5P"
            ]
            exact = [
                row for row in exact_probe_rows
                if row.get("target") == target and row.get("block") == block
                and row.get("feature_set") == "current_C5P_plus_NVe_Vort_Bphi"
            ]
            if current and exact and float(current[0]["normalized_RMSE"]) > 0.0:
                reduction = (
                    float(current[0]["normalized_RMSE"])
                    - float(exact[0]["normalized_RMSE"])
                ) / float(current[0]["normalized_RMSE"])
                passed += int(reduction >= 0.10)
        if passed == 3:
            exact_probe_success.append(target)
    state_families = sorted(
        {str(target) for target, _ in state_success}
        | {probe_target_family(target) for target in exact_probe_success}
    )
    state_ratio = len(state_families) / 2.0
    state_tier = "strong" if len(state_families) >= 2 else "moderate" if state_ratio >= 0.5 else "weak" if state_families else "none"
    evidence.append(
        {
            "explanation": "insufficient or incorrect retained state",
            "tier": state_tier,
            "normalized_effect": state_ratio,
            "summary": f"{len(state_families)} target families pass the all-three-block >=10% exact-state improvement rule ({len(state_success)} by causal neighbors; {len(exact_probe_success)} scalar targets by normalized probe RMSE).",
        }
    )

    history = [row for row in context_rows if row.get("analysis") == "history_probe" and row.get("probe") == "context_ridge"]
    history_success = []
    for target in {str(row["target"]) for row in history}:
        passed = 0
        effects = []
        for block in ("V00", "V01", "V02"):
            current = [row for row in history if row.get("target") == target and row.get("block") == block and row.get("feature_set") == "current_C5P_history_matched"]
            delayed = [row for row in history if row.get("target") == target and row.get("block") == block and row.get("feature_set") == "C5P_delays_1_2_4_8_16"]
            if current and delayed:
                r2_gain = float(delayed[0]["R2"]) - float(current[0]["R2"])
                rmse_gain = (float(current[0]["normalized_RMSE"]) - float(delayed[0]["normalized_RMSE"])) / float(current[0]["normalized_RMSE"])
                effects.append(max(r2_gain / 0.05, rmse_gain / 0.10))
                passed += int(r2_gain >= 0.05 or rmse_gain >= 0.10)
        if passed == 3:
            history_success.append((target, min(effects)))
    history_families = sorted({probe_target_family(target) for target, _ in history_success})
    history_ratio = len(history_families) / 2.0
    history_tier = "strong" if len(history_families) >= 2 else "moderate" if history_ratio >= 0.5 else "weak" if history_families else "none"
    evidence.append(
        {
            "explanation": "history-dependent hidden state",
            "tier": history_tier,
            "normalized_effect": history_ratio,
            "summary": f"{len(history_success)} scalar targets spanning {len(history_families)} target families improve at the frozen history threshold in all three validation blocks.",
        }
    )

    relevant_ess = [
        row for row in ess_rows
        if row.get("region") == "training" and row.get("detrended") is False
        and row.get("method") == "Geyer_initial_positive_pair_sequence"
        and ("residual" in str(row.get("observable")) or "transport" in str(row.get("observable")))
    ]
    low_ess_fraction = float(np.mean([float(row["effective_sample_size"]) < 20 for row in relevant_ess])) if relevant_ess else 0.0
    def late_curve(prefix: int) -> float:
        values = [
            float(row["value"]) for row in learning_rows
            if int(row["source_prefix_targets"]) == prefix
            and str(row["target_block"]).startswith("V") and int(row["rank"]) in (8, 16, 32)
        ]
        return float(np.mean(values))
    late_gain = late_curve(420) - late_curve(378)
    sampling_strong = low_ess_fraction >= 0.5 and late_gain > 0.02
    sampling_ratio = min(low_ess_fraction / 0.5, 2.0) * 0.5 + max(late_gain / 0.02, 0.0) * 0.5
    sampling_tier = "strong" if sampling_strong else "moderate" if low_ess_fraction >= 0.5 or late_gain > 0.02 else "weak" if low_ess_fraction > 0 else "none"
    evidence.append(
        {
            "explanation": "insufficient effective sample size",
            "tier": sampling_tier,
            "normalized_effect": sampling_ratio,
            "summary": f"{low_ess_fraction:.3f} of material residual/transport observables have ESS<20; 378-to-420 prefix mean capture gain={late_gain:.4f}.",
        }
    )

    substantive = [item for item in evidence if item["explanation"] != "localized or multiscale representation advantage"]
    any_moderate = any(TIER_ORDER[item["tier"]] >= TIER_ORDER["moderate"] for item in substantive)
    evidence.append(
        {
            "explanation": "unexplained failure",
            "tier": "none" if any_moderate else "moderate",
            "normalized_effect": 0.0 if any_moderate else 1.0,
            "summary": "Reserved for the case in which no preregistered mechanism reaches moderate evidence.",
        }
    )

    # Required explanation list excludes the auxiliary representation row; it
    # remains evidence for the action mapping and is reported after ranking.
    required_order = {
        "invalid/nonstationary interval": 0,
        "coherent transport in an inappropriate Eulerian representation": 1,
        "codec or predictor non-equivariance": 2,
        "forecast-state-dependent covariance": 3,
        "insufficient or incorrect retained state": 4,
        "history-dependent hidden state": 5,
        "insufficient effective sample size": 6,
        "unexplained failure": 7,
    }
    ranked = sorted(
        [item for item in evidence if item["explanation"] in required_order],
        key=lambda item: (-TIER_ORDER[item["tier"]], -float(item["normalized_effect"]), required_order[item["explanation"]]),
    )
    lookup = {item["explanation"]: item for item in evidence}
    if lookup["invalid/nonstationary interval"]["tier"] == "strong":
        action = "repair interval/conditioning"
    elif lookup["coherent transport in an inappropriate Eulerian representation"]["tier"] == "strong" or lookup["codec or predictor non-equivariance"]["tier"] == "strong":
        action = "build an equivariant transport-plus-innovation model"
    elif (
        lookup["forecast-state-dependent covariance"]["tier"] == "strong"
        or lookup["history-dependent hidden state"]["tier"] == "strong"
        or lookup["localized or multiscale representation advantage"]["tier"] == "strong"
    ):
        action = "strengthen context-conditioned multiscale residual generation"
    elif lookup["insufficient effective sample size"]["tier"] == "strong":
        action = "request independent Hermes restarts"
    else:
        action = "stop model development and write the benchmark/failure result"
    return ranked + [lookup["localized or multiscale representation advantage"]], action


def decision_memo(evidence: Sequence[Mapping[str, Any]], action: str, *, run_record: Mapping[str, Any]) -> str:
    required = [item for item in evidence if item["explanation"] != "localized or multiscale representation advantage"]
    auxiliary = next(item for item in evidence if item["explanation"] == "localized or multiscale representation advantage")
    lines = [
        "A single, fixed, condition-independent, global linear residual distribution learned from adjacent 85604 training frames does not describe later 85604 residuals well.",
        "",
        "# Paper 0 Phase 3.5 decision memo",
        "",
        f"**Authoritative job:** `{run_record['slurm_job_id']}` at commit `{run_record['paper0_commit']}`",
        "",
        "**Scope:** simulation 85604 only. The guard was not read. Simulation 85606 remains unopened. This phase trained no production neural model and performed no assimilation.",
        "",
        "## What this memo does and does not conclude",
        "",
        "Phase 3.5 localizes why K4's fixed global linear residual model failed. It does not say that stochastic emulation of 85604 is impossible, and it does not reinterpret K4 as a test of FGN, PDE-Refiner, or diffusion.",
        "",
        "## Ranked explanations",
        "",
    ]
    for rank, item in enumerate(required, 1):
        lines.extend(
            [
                f"{rank}. **{item['explanation']} — {item['tier']} evidence.**",
                f"   {item['summary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Representation companion result",
            "",
            f"**{auxiliary['tier']} evidence:** {auxiliary['summary']}",
            "",
            "The representation comparison is used to decide whether localization or multiscale structure transfers chronologically. It is not an architecture competition and no target-block energy selected its coefficients.",
            "",
            "## Interpretation",
            "",
            "The decision follows the preregistered priority: state/protocol validity, stationarity, coherent transport/equivariance, context/history, and only then stochastic capacity. Truth-assisted shifts and truth-projected residual coefficients are explicitly nondeployable diagnostic upper bounds.",
            "",
            f"Recommended next action: {action}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    start_wall = time.monotonic()
    checkout = verify_checkout(args.paper0_commit)
    protocol = load_phase3_5_protocol(args.manifest, root=ROOT)
    manifest_hash = sha256_path(args.manifest)
    clarifications = {
        ROOT / "paper0/PHASE3_5_PROTOCOL_AMENDMENT_2026-08-19A.md":
            "4846c2f63a5dd634ead314517516e6727ddff5e85c6779cc114e35300db73d75",
        ROOT / "paper0/PHASE3_5_PROTOCOL_AMENDMENT_2026-08-19B.md":
            "0870d968ae898ed4058dfd1eec0b6a8ecd1d1fd10c4de5cad6c1e2490d61d3e0",
    }
    for clarification, expected_hash in clarifications.items():
        if sha256_path(clarification) != expected_hash:
            raise RuntimeError(
                f"Phase 3.5 execution clarification hash differs: {clarification.name}"
            )
    native_result_path = ROOT / NATIVE_RESULT_RELATIVE
    geometry_manifest_path = ROOT / GEOMETRY_MANIFEST_RELATIVE
    if sha256_path(native_result_path) != NATIVE_RESULT_SHA256:
        raise RuntimeError("native-state compact result hash differs")
    if sha256_path(geometry_manifest_path) != GEOMETRY_MANIFEST_SHA256:
        raise RuntimeError("geometry manifest hash differs")
    output = Path(args.output_directory)
    scratch = Path(args.scratch_directory)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    scratch.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    torch.set_num_threads(min(24, int(os.environ.get("SLURM_CPUS_PER_TASK", "24"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import wandb

    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        job_type="phase3_5_85604_cause_localization",
        id=f"p35-{args.slurm_job_id}",
        resume="never",
        mode="online",
        config={
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "development_run": "85604",
            "held_out_access": False,
            "production_model_training": False,
            "manifest_sha256": manifest_hash,
            "protocol_sha256": protocol.record["protocol"]["sha256"],
            "clarification_sha256s": {
                path.name: expected_hash
                for path, expected_hash in clarifications.items()
            },
            "zperiod": 5,
            "mode_mapping": "n=5k",
        },
        settings=wandb.Settings(init_timeout=120),
    )
    if wandb_run is None or not wandb_run.url:
        raise RuntimeError("online W&B initialization failed")
    progress("phase3_5_start", wandb_url=wandb_run.url, commit=args.paper0_commit)
    stage_times: dict[str, float] = {}

    def stage(name: str, started: float) -> None:
        elapsed = time.monotonic() - started
        stage_times[name] = elapsed
        wandb_run.log({f"stage_seconds/{name}": elapsed, "elapsed_seconds": time.monotonic() - start_wall})
        progress("stage_complete", name=name, seconds=elapsed)

    try:
        current = time.monotonic()
        artifacts = verify_primary_artifacts()
        catalog = load_official_catalog(ARTIFACT_ROOT)
        geometry_manifest = load_strict_json(geometry_manifest_path)
        geometry = load_transport_geometry(geometry_path=GEOMETRY, geometry_manifest=geometry_manifest)
        native_catalog = NativeTruthCatalog(load_strict_json(native_result_path))
        training_targets = tuple(range(2, 432))
        validation_targets = tuple(range(498, 624))
        targets = training_targets + validation_targets
        stage("input_verification", current)

        current = time.monotonic()
        training_all_std = load_c5_frames(catalog, range(0, 432), physical=False)
        validation_all_std = load_c5_frames(catalog, range(496, 624), physical=False)
        training_all_exact, training_all_boundary = load_exact_frames(catalog, range(0, 432))
        validation_all_exact, validation_all_boundary = load_exact_frames(catalog, range(496, 624))
        training_truth = training_all_std[2:]
        validation_truth = validation_all_std[2:]
        training_context = training_all_std[1:431]
        validation_context = validation_all_std[1:127]
        training_previous = training_all_std[:430]
        validation_previous = validation_all_std[:126]
        training_exact_target = training_all_exact[2:]
        validation_exact_target = validation_all_exact[2:]
        training_exact_context = training_all_exact[1:431]
        validation_exact_context = validation_all_exact[1:127]
        training_boundary_target = training_all_boundary[2:]
        validation_boundary_target = validation_all_boundary[2:]
        training_boundary_context = training_all_boundary[1:431]
        validation_boundary_context = validation_all_boundary[1:127]
        training_h1, validation_h1 = load_h1_forecasts(
            training_targets=training_targets, validation_targets=validation_targets
        )
        artifacts.extend(
            {
                "name": f"model_dataset_shard_{shard.index}",
                "path": str(shard.path.resolve(strict=True)),
                "sha256": shard.sha256,
                "bytes": shard.path.stat().st_size,
            }
            for shard in catalog.shards
        )
        training_residual, validation_residual, axisymmetric_bias = corrected_h1_residuals(
            training_truth, training_h1, validation_truth, validation_h1
        )
        physical_training_truth = decode_c5(catalog, training_truth)
        physical_validation_truth = decode_c5(catalog, validation_truth)
        stage("data_load_and_residual_construction", current)

        current = time.monotonic()
        native_training = native_catalog.read(2, 432)
        native_validation = native_catalog.read(498, 624)
        training_truth_transport = transport_from_native(native_training, geometry)
        validation_truth_transport = transport_from_native(native_validation, geometry)
        training_h1_transport = transport_from_standardized(training_h1, catalog, geometry)
        validation_h1_transport = transport_from_standardized(validation_h1, catalog, geometry)
        training_transport_error = subtract_transport(training_h1_transport, training_truth_transport)
        validation_transport_error = subtract_transport(validation_h1_transport, validation_truth_transport)
        truth_transport_all = concatenate_transport(training_truth_transport, validation_truth_transport)
        transport_error_all = concatenate_transport(training_transport_error, validation_transport_error)
        stage("authoritative_transport", current)

        current = time.monotonic()
        field_rms = training_field_rms(training_all_std)
        translation_training, aligned_training_h1, transported_training, recent_training = run_translation_diagnostics(
            targets=training_targets,
            previous_states=training_previous,
            context_states=training_context,
            truth=training_truth,
            h1=training_h1,
            field_rms=field_rms,
        )
        translation_validation, aligned_validation_h1, transported_validation, recent_validation = run_translation_diagnostics(
            targets=validation_targets,
            previous_states=validation_previous,
            context_states=validation_context,
            truth=validation_truth,
            h1=validation_h1,
            field_rms=field_rms,
        )
        translation_rows = translation_training + translation_validation
        aligned_training_residual, aligned_validation_residual, aligned_bias = corrected_h1_residuals(
            training_truth, aligned_training_h1, validation_truth, aligned_validation_h1
        )
        if not np.allclose(aligned_bias, axisymmetric_bias, rtol=2e-6, atol=2e-6):
            raise RuntimeError("a shared z shift unexpectedly changed the axisymmetric bias")
        aligned_training_transport = transport_from_standardized(aligned_training_h1, catalog, geometry)
        aligned_validation_transport = transport_from_standardized(aligned_validation_h1, catalog, geometry)
        transported_training_transport = transport_from_standardized(transported_training, catalog, geometry)
        transported_validation_transport = transport_from_standardized(transported_validation, catalog, geometry)
        append_translation_physics_rows(
            translation_rows,
            targets=targets,
            truth=np.concatenate((training_truth, validation_truth)),
            candidates={
                "H1_unaligned": np.concatenate((training_h1, validation_h1)),
                "H1_truth_assisted_shared_shift": np.concatenate((aligned_training_h1, aligned_validation_h1)),
                "persistence_unshifted": np.concatenate((training_context, validation_context)),
                "transported_persistence_truth_assisted_shared_shift": np.concatenate((transported_training, transported_validation)),
            },
            truth_transport=truth_transport_all,
            candidate_transport={
                "H1_unaligned": concatenate_transport(training_h1_transport, validation_h1_transport),
                "H1_truth_assisted_shared_shift": concatenate_transport(aligned_training_transport, aligned_validation_transport),
                "persistence_unshifted": concatenate_transport(
                    transport_from_standardized(training_context, catalog, geometry),
                    transport_from_standardized(validation_context, catalog, geometry),
                ),
                "transported_persistence_truth_assisted_shared_shift": concatenate_transport(
                    transported_training_transport, transported_validation_transport
                ),
            },
        )
        stage("translation_oracles", current)

        current = time.monotonic()
        all_residual = np.concatenate((training_residual, validation_residual))
        all_aligned_residual = np.concatenate((aligned_training_residual, aligned_validation_residual))
        residual_raw_gram = raw_sample_gram(all_residual, torch_device=device, row_batch=32)
        aligned_raw_gram = raw_sample_gram(all_aligned_residual, torch_device=device, row_batch=32)
        stage("residual_gram_matrices", current)

        current = time.monotonic()
        stationarity_rows, series_by_object = build_stationarity_rows(
            protocol=protocol,
            targets=targets,
            physical_truth=np.concatenate((physical_training_truth, physical_validation_truth)),
            residual=all_residual,
            exact_state=np.concatenate((training_exact_target, validation_exact_target)),
            boundary=np.concatenate((training_boundary_target, validation_boundary_target)),
            truth_transport=truth_transport_all,
            error_transport=transport_error_all,
        )
        contrasts = stationarity_contrasts(protocol, targets, series_by_object)
        training_series: dict[str, np.ndarray] = {}
        validation_series: dict[str, np.ndarray] = {}
        for object_name, mapping in series_by_object.items():
            for metric, values in mapping.items():
                name = f"{object_name}.{metric}"
                series = np.asarray(values)
                if metric.endswith(".phase_radians"):
                    # Wrapped angles do not admit an ordinary linear ACF.  The
                    # sine/cosine components retain circular information
                    # without manufacturing jumps at +/-pi.
                    training_series[f"{name}.sine"] = np.sin(series[:430])
                    training_series[f"{name}.cosine"] = np.cos(series[:430])
                    validation_series[f"{name}.sine"] = np.sin(series[430:])
                    validation_series[f"{name}.cosine"] = np.cos(series[430:])
                else:
                    training_series[name] = series[:430]
                    validation_series[name] = series[430:]
        ess_rows = build_ess_rows(
            training_series=training_series,
            validation_series=validation_series,
            residual_training=training_residual,
            residual_raw_gram=residual_raw_gram,
        )
        transfer_rows = build_block_transfer_rows(
            protocol=protocol,
            targets=targets,
            residual=all_residual,
            raw_gram=residual_raw_gram,
            transport_error=transport_error_all,
        )
        learning_rows = build_learning_curve_rows(
            residual_raw_gram=residual_raw_gram, training_count=430
        )
        stage("stationarity_ess_transfer", current)

        current = time.monotonic()
        k4_rows = run_aligned_k4_ladder(
            training_residual=aligned_training_residual,
            validation_residual=aligned_validation_residual,
            aligned_validation_h1=aligned_validation_h1,
            axisymmetric_bias=axisymmetric_bias,
            raw_gram=aligned_raw_gram,
            validation_truth=validation_truth,
            truth_transport=validation_truth_transport,
            catalog=catalog,
            geometry=geometry,
            device=device,
        )
        translation_rows.extend(k4_rows)
        stage("aligned_K4_ladder", current)

        current = time.monotonic()
        representative_targets = tuple((block.start + block.stop - 1) // 2 for block in protocol.blocks)
        target_map = {target: index for index, target in enumerate(targets)}
        representative_indices = np.asarray([target_map[target] for target in representative_targets])
        all_context = np.concatenate((training_context, validation_context))
        all_truth = np.concatenate((training_truth, validation_truth))
        equivariance_rows, equivariance_curves = audit_frozen_h1_equivariance(
            all_context[representative_indices, None],
            all_truth[representative_indices],
            target_frames=representative_targets,
            device=device,
        )
        stage("equivariance", current)

        current = time.monotonic()
        representation_rows = run_representation_audit(
            protocol=protocol,
            training_targets=training_targets,
            validation_targets=validation_targets,
            training_residual=training_residual,
            validation_residual=validation_residual,
            aligned_training_residual=aligned_training_residual,
            aligned_validation_residual=aligned_validation_residual,
            training_h1=training_h1,
            validation_h1=validation_h1,
            aligned_training_h1=aligned_training_h1,
            aligned_validation_h1=aligned_validation_h1,
            axisymmetric_bias=axisymmetric_bias,
            training_truth_transport=training_truth_transport,
            validation_truth_transport=validation_truth_transport,
            catalog=catalog,
            geometry=geometry,
        )
        representation_rows.extend(learning_rows)
        stage("matched_representation_audit", current)

        current = time.monotonic()
        def translation_target_arrays(rows: Sequence[Mapping[str, Any]], target_order: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
            shifts = {}
            gains = {}
            for row in rows:
                if row.get("estimator") == "H1_prediction_to_truth_oracle" and row.get("field") == "shared_whitened_multichannel":
                    shifts[int(row["target_frame"])] = float(row["signed_integer_shift"])
                if row.get("estimator") == "alignment_gain":
                    gains[int(row["target_frame"])] = float(row["alignment_fractional_energy_reduction"])
            return (
                np.asarray([shifts[target] for target in target_order]),
                np.asarray([gains[target] for target in target_order]),
            )
        training_shift, training_gain = translation_target_arrays(translation_rows, training_targets)
        validation_shift, validation_gain = translation_target_arrays(translation_rows, validation_targets)
        training_target_matrix, target_names = build_probe_targets(
            residual=training_residual,
            truth=training_truth,
            h1=training_h1,
            transport_error=training_transport_error,
            oracle_shift=training_shift,
            alignment_gain=training_gain,
        )
        validation_target_matrix, validation_target_names = build_probe_targets(
            residual=validation_residual,
            truth=validation_truth,
            h1=validation_h1,
            transport_error=validation_transport_error,
            oracle_shift=validation_shift,
            alignment_gain=validation_gain,
        )
        if target_names != validation_target_names:
            raise RuntimeError("probe target schema differs between chronological regions")
        context_rows = run_context_and_state_probes(
            protocol=protocol,
            training_targets=training_targets,
            validation_targets=validation_targets,
            training_context=training_context,
            validation_context=validation_context,
            training_recent_features=recent_training,
            validation_recent_features=recent_validation,
            training_exact_context=training_exact_context,
            validation_exact_context=validation_exact_context,
            training_boundary_context=training_boundary_context,
            validation_boundary_context=validation_boundary_context,
            training_target_matrix=training_target_matrix,
            validation_target_matrix=validation_target_matrix,
            target_names=target_names,
            training_all_c5=training_all_std,
            validation_all_c5=validation_all_std,
            training_all_exact=training_all_exact,
            validation_all_exact=validation_all_exact,
            training_all_boundary=training_all_boundary,
            validation_all_boundary=validation_all_boundary,
        )
        stage("context_state_history_probes", current)

        current = time.monotonic()
        validation_context_map = {
            target: validation_context[index] for index, target in enumerate(validation_targets)
        }
        validation_h1_map = {
            target: validation_h1[index] for index, target in enumerate(validation_targets)
        }
        b5_rows, b5_record = run_b5_context_shuffle(
            validation_context_by_target=validation_context_map,
            validation_h1_by_target=validation_h1_map,
            device=device,
            member_metric_callback=b5_transport_shuffle_callback(catalog, geometry),
        )
        context_rows.extend({"analysis": "B5_context_shuffle", **row} for row in b5_rows)
        stage("B5_context_shuffle", current)

        current = time.monotonic()
        atomic_csv(output / "stationarity_summary.csv", stationarity_rows)
        atomic_csv(output / "stationarity_contrasts.csv", contrasts)
        atomic_csv(output / "effective_sample_size.csv", ess_rows)
        atomic_csv(output / "block_transfer_matrix.csv", transfer_rows)
        atomic_csv(output / "translation_diagnostics.csv", translation_rows)
        atomic_csv(output / "equivariance_audit.csv", equivariance_rows)
        atomic_csv(output / "representation_transfer.csv", representation_rows)
        atomic_csv(output / "context_dependence.csv", context_rows)
        atomic_json(output / "equivariance_curves.json", equivariance_curves)
        atomic_json(output / "B5_context_shuffle_provenance.json", b5_record)
        atomic_npz(
            output / "sufficient_statistics.npz",
            residual_raw_gram=residual_raw_gram,
            aligned_residual_raw_gram=aligned_raw_gram,
            translation_training_field_rms=field_rms,
            axisymmetric_bias=axisymmetric_bias,
            target_frames=np.asarray(targets),
        )
        figures = make_plots(
            output=output,
            stationarity_rows=stationarity_rows,
            ess_rows=ess_rows,
            transfer_rows=transfer_rows,
            translation_rows=translation_rows,
            equivariance_rows=equivariance_rows,
            representation_rows=representation_rows,
            context_rows=context_rows,
        )
        captions = "# Phase 3.5 figure captions\n\n" + "\n\n".join(
            f"## {item['stem']}\n\n{item['caption']}" for item in figures
        ) + "\n"
        atomic_text(output / "figure_captions.md", captions)
        evidence, action = classify_evidence(
            contrasts=contrasts,
            ess_rows=ess_rows,
            learning_rows=learning_rows,
            translation_rows=translation_rows,
            equivariance_rows=equivariance_rows,
            equivariance_curves=equivariance_curves,
            representation_rows=representation_rows,
            context_rows=context_rows,
        )
        atomic_json(output / "evidence_ranking.json", {"evidence": evidence, "recommended_next_action": action})
        run_record = {"slurm_job_id": args.slurm_job_id, "paper0_commit": args.paper0_commit}
        memo = decision_memo(evidence, action, run_record=run_record)
        atomic_text(output / "PHASE3_5_DECISION_MEMO.md", memo)
        stage("tables_figures_decision", current)

        output_records = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "run_manifest.json":
                output_records.append(
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_path(path)}
                )
        manifest_record = {
            "schema_version": 1,
            "status": "completed_phase3_5_without_architecture_training_or_downstream_opening",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "production_model_training_performed": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "paper0_commit": args.paper0_commit,
            "checkout": checkout,
            "slurm_job_id": args.slurm_job_id,
            "command": [sys.executable, *sys.argv],
            "started_utc": datetime.fromtimestamp(time.time() - (time.monotonic() - start_wall), timezone.utc).isoformat(),
            "completed_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - start_wall,
            "stage_seconds": stage_times,
            "host": platform.node(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "CUDA_device": torch.cuda.get_device_name(0),
            "TF32_allowed": False,
            "seeds": protocol.record["seeds"],
            "protocol": {
                "manifest": {"path": str(args.manifest), "sha256": manifest_hash},
                "primary": protocol.record["protocol"],
                "clarifications": protocol.record["clarifying_amendments"],
            },
            "inputs": artifacts + [
                {"name": "native_state_result", "path": str(native_result_path), "sha256": NATIVE_RESULT_SHA256},
                {"name": "geometry_manifest", "path": str(geometry_manifest_path), "sha256": GEOMETRY_MANIFEST_SHA256},
            ],
            "data_ranges": protocol.record["data"],
            "wandb": {"url": wandb_run.url, "id": wandb_run.id, "online": True, "large_artifacts_uploaded": False},
            "outputs": output_records,
            "recommended_next_action": action,
            "recommended_action_automatically_authorized": False,
            "85606_access_automatically_authorized": False,
        }
        manifest_path = atomic_json(output / "run_manifest.json", manifest_record)
        wandb_run.summary.update(
            {
                "status": "completed",
                "recommended_next_action": action,
                "elapsed_seconds": manifest_record["elapsed_seconds"],
                "output_file_count": len(output_records) + 1,
                "run_manifest_sha256": sha256_path(manifest_path),
                "held_out_85606_read": False,
                "guard_frames_read": False,
            }
        )
        wandb_run.finish(exit_code=0)
        progress(
            "phase3_5_complete",
            output_directory=str(output),
            run_manifest_sha256=sha256_path(manifest_path),
            recommended_next_action=action,
        )
    except BaseException:
        wandb_run.summary.update(
            {
                "status": "failed_without_authoritative_decision",
                "held_out_85606_read": False,
                "guard_frames_read": False,
            }
        )
        wandb_run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
