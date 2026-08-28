#!/usr/bin/env python3
"""Run frozen S0 simultaneous spatial reconstruction on old 85604 only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from tcv_diagnostics.matched_o1_transport import load_transport_geometry
from tcv_diagnostics.model_data import (
    assert_development_path,
    load_strict_json,
    sha256_file,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.spatial_s0 import (
    DISTANCE_BIN_EDGES_M,
    FOOTPRINT_CENTERS,
    INTERNAL_FIT_INTERVAL,
    INTERNAL_GUARD_INTERVAL,
    INTERNAL_TUNE_INTERVAL,
    RIDGE_LAMBDAS,
    S0_FIELDS,
    TRAIN_INTERVAL,
    VALIDATION_INTERVAL,
    VOLUME_SHAPE,
    DualRidgeKernel,
    basic_metrics,
    build_fixed_footprints,
    choose_regularization,
    group_footprints,
    minimum_cylindrical_distance_to_observations,
    observe_density,
    select_median_hero_frame,
    toroidal_mode_metrics,
)


PROTOCOL = "paper0/protocol/PHYSICS_FIRST_SPATIAL_S0_PROTOCOL_2026-08-28.md"
OUTPUT_X_SLAB = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", default="local")
    return parser.parse_args()


class OfficialS0Source:
    """Selected-field slab access without materializing the joint target."""

    def __init__(self, artifact_root: Path) -> None:
        assert_development_path(artifact_root)
        self.catalog = load_official_catalog(artifact_root)
        self.handles: dict[Path, h5py.File] = {}

    def _handle(self, path: Path) -> h5py.File:
        handle = self.handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self.handles[path] = handle
        return handle

    def standardized_slab(
        self,
        field: str,
        frames: Sequence[int],
        x_slice: slice,
    ) -> np.ndarray:
        if field not in S0_FIELDS:
            raise ValueError(f"field {field!r} is not an S0 field")
        indices = tuple(int(frame) for frame in frames)
        if not indices or indices != tuple(range(indices[0], indices[-1] + 1)):
            raise ValueError("S0 source frames must be nonempty and contiguous")
        allowed = (
            TRAIN_INTERVAL[0] <= indices[0] and indices[-1] < TRAIN_INTERVAL[1]
        ) or (
            VALIDATION_INTERVAL[0] <= indices[0]
            and indices[-1] < VALIDATION_INTERVAL[1]
        )
        if not allowed:
            raise ValueError("S0 source request leaves frozen train/validation ranges")
        xs = np.arange(*x_slice.indices(VOLUME_SHAPE[0]), dtype=np.int64)
        if xs.size == 0 or not np.array_equal(xs, np.arange(xs[0], xs[-1] + 1)):
            raise ValueError("x_slice must select a nonempty contiguous slab")
        self.catalog.verify_consumed_frames(indices)
        output = np.empty(
            (len(indices), xs.size, VOLUME_SHAPE[1], VOLUME_SHAPE[2]),
            dtype=np.float32,
        )
        destination = 0
        for shard in self.catalog.shards:
            start = max(indices[0], shard.start)
            stop = min(indices[-1] + 1, shard.stop)
            if start >= stop:
                continue
            local_start = start - shard.start
            local_stop = stop - shard.start
            raw = np.asarray(
                self._handle(shard.path)[f"fields/{field}"][
                    local_start:local_stop, xs[0] : xs[-1] + 1, :, :
                ]
            )
            encoded = self.catalog.normalization.records[field].encode(raw)
            count = stop - start
            output[destination : destination + count] = encoded
            destination += count
        if destination != len(indices):
            raise RuntimeError("S0 source failed to cover requested frames")
        return output

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise ValueError(f"refusing to write empty CSV {path}")
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _frame_range(interval: tuple[int, int]) -> tuple[int, ...]:
    return tuple(range(*interval))


def _load_geometry(
    geometry_path: Path,
    manifest_path: Path,
) -> tuple[Any, np.ndarray, np.ndarray, Mapping[str, Any]]:
    assert_development_path(geometry_path)
    assert_development_path(manifest_path)
    manifest = load_strict_json(manifest_path)
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=manifest,
    )
    grid = manifest["grid"]
    offset = int(grid["model_x_to_grid_x_offset"])
    with h5py.File(geometry_path, "r") as handle:
        radius = np.asarray(handle["Rxy"][offset : offset + 64], dtype=np.float64)
        vertical = np.asarray(handle["Zxy"][offset : offset + 64], dtype=np.float64)
    if radius.shape != VOLUME_SHAPE[:2] or vertical.shape != radius.shape:
        raise ValueError("geometry cell-center arrays differ from model shape")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(vertical)):
        raise ValueError("geometry cell centers contain non-finite values")
    return geometry, radius, vertical, manifest


def _training_climatology(source: OfficialS0Source) -> np.ndarray:
    frames = _frame_range(TRAIN_INTERVAL)
    climatology = np.empty((len(S0_FIELDS), *VOLUME_SHAPE), dtype=np.float32)
    for field_index, field in enumerate(S0_FIELDS):
        for x0 in range(0, VOLUME_SHAPE[0], OUTPUT_X_SLAB):
            x1 = min(x0 + OUTPUT_X_SLAB, VOLUME_SHAPE[0])
            values = source.standardized_slab(field, frames, slice(x0, x1))
            climatology[field_index, x0:x1] = np.mean(
                values, axis=0, dtype=np.float64
            ).astype(np.float32)
    return climatology


def _observations(
    source: OfficialS0Source,
    climatology_ne: np.ndarray,
    frames: Sequence[int],
    footprints: Sequence[Any],
) -> np.ndarray:
    result = np.empty((len(frames), len(footprints)), dtype=np.float64)
    block = 16
    for start in range(0, len(frames), block):
        stop = min(start + block, len(frames))
        values = source.standardized_slab(
            "Ne", frames[start:stop], slice(0, VOLUME_SHAPE[0])
        )
        values -= climatology_ne[None]
        result[start:stop] = observe_density(values, footprints)
    return result


def _fit_and_tune(
    source: OfficialS0Source,
    climatology: np.ndarray,
    fit_inputs: np.ndarray,
    tune_inputs: np.ndarray,
    fit_c: np.ndarray,
    tune_c: np.ndarray,
    strict_xy: np.ndarray,
) -> tuple[float, tuple[dict[str, float], ...]]:
    kernels = {value: DualRidgeKernel.fit(fit_inputs, value) for value in RIDGE_LAMBDAS}
    state_sse = {
        value: np.zeros(len(S0_FIELDS), dtype=np.float64) for value in RIDGE_LAMBDAS
    }
    state_count = np.zeros(len(S0_FIELDS), dtype=np.int64)
    fit_frames = _frame_range(INTERNAL_FIT_INTERVAL)
    tune_frames = _frame_range(INTERNAL_TUNE_INTERVAL)
    for field_index, field in enumerate(S0_FIELDS):
        for x0 in range(0, VOLUME_SHAPE[0], OUTPUT_X_SLAB):
            x1 = min(x0 + OUTPUT_X_SLAB, VOLUME_SHAPE[0])
            fit_target = source.standardized_slab(
                field, fit_frames, slice(x0, x1)
            ).astype(np.float64)
            tune_target = source.standardized_slab(
                field, tune_frames, slice(x0, x1)
            ).astype(np.float64)
            mean = climatology[field_index, x0:x1].astype(np.float64)
            fit_target -= mean[None]
            tune_target -= mean[None]
            fit_flat = fit_target.reshape(len(fit_frames), -1)
            local_mask = strict_xy[x0:x1]
            count = int(len(tune_frames) * np.sum(local_mask) * VOLUME_SHAPE[2])
            state_count[field_index] += count
            truth_masked = tune_target[:, local_mask, :]
            for value, kernel in kernels.items():
                prediction = kernel.predict_equivalent_dual(
                    tune_inputs, fit_flat
                ).reshape(tune_target.shape)
                error = prediction[:, local_mask, :] - truth_masked
                state_sse[value][field_index] += float(np.sum(error * error))
    records: list[dict[str, float]] = []
    for value, kernel in kernels.items():
        c_prediction = kernel.predict_equivalent_dual(tune_inputs, fit_c)
        c_rmse = float(np.sqrt(np.mean((c_prediction - tune_c) ** 2)))
        field_rmse = np.sqrt(state_sse[value] / state_count)
        records.append(
            {
                "regularization": value,
                "equal_field_full_state_rmse": float(np.mean(field_rmse)),
                "heldout_c_rmse": c_rmse,
            }
        )
    return choose_regularization(records)


def _masked_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    boolean = np.asarray(mask, dtype=bool)
    if boolean.shape == VOLUME_SHAPE[:2]:
        return basic_metrics(truth[:, boolean, :], prediction[:, boolean, :])
    if boolean.shape == VOLUME_SHAPE:
        return basic_metrics(truth[:, boolean], prediction[:, boolean])
    raise ValueError("metric mask must have shape [x,y] or [x,y,z]")


def _paired_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    zero = _masked_metrics(truth, np.zeros_like(truth), mask)
    ridge = _masked_metrics(truth, prediction, mask)
    zero_mse = float(zero["rmse"]) ** 2
    ridge_mse = float(ridge["rmse"]) ** 2
    zero["relative_mse_skill_vs_zero"] = 0.0
    ridge["relative_mse_skill_vs_zero"] = (
        1.0 - ridge_mse / zero_mse if zero_mse > 0.0 else 0.0
    )
    return zero, ridge


def _evaluate_full_state(
    source: OfficialS0Source,
    climatology: np.ndarray,
    kernel: DualRidgeKernel,
    train_inputs: np.ndarray,
    validation_inputs: np.ndarray,
    strict_xy: np.ndarray,
    regions: Mapping[str, np.ndarray],
    distance_m: np.ndarray,
    hero_frame: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
]:
    train_frames = _frame_range(TRAIN_INTERVAL)
    validation_frames = _frame_range(VALIDATION_INTERVAL)
    hero_position = hero_frame - VALIDATION_INTERVAL[0]
    field_records: list[dict[str, Any]] = []
    region_records: list[dict[str, Any]] = []
    distance_records: list[dict[str, Any]] = []
    mode_records: list[dict[str, Any]] = []
    hero_truth = np.empty((len(S0_FIELDS), *VOLUME_SHAPE), dtype=np.float32)
    hero_prediction = np.empty_like(hero_truth)

    for field_index, field in enumerate(S0_FIELDS):
        truth = np.empty((len(validation_frames), *VOLUME_SHAPE), dtype=np.float32)
        prediction = np.empty_like(truth)
        for x0 in range(0, VOLUME_SHAPE[0], OUTPUT_X_SLAB):
            x1 = min(x0 + OUTPUT_X_SLAB, VOLUME_SHAPE[0])
            fit_target = source.standardized_slab(
                field, train_frames, slice(x0, x1)
            ).astype(np.float64)
            validation_target = source.standardized_slab(
                field, validation_frames, slice(x0, x1)
            ).astype(np.float64)
            mean = climatology[field_index, x0:x1].astype(np.float64)
            fit_target -= mean[None]
            validation_target -= mean[None]
            predicted = kernel.predict_equivalent_dual(
                validation_inputs,
                fit_target.reshape(len(train_frames), -1),
            ).reshape(validation_target.shape)
            truth[:, x0:x1] = validation_target.astype(np.float32)
            prediction[:, x0:x1] = predicted.astype(np.float32)

        hero_truth[field_index] = truth[hero_position]
        hero_prediction[field_index] = prediction[hero_position]
        full_zero, full_ridge = _paired_metrics(truth, prediction, strict_xy)
        for method, metrics in (
            ("zero_fluctuation", full_zero),
            ("ridge", full_ridge),
        ):
            field_records.append({"method": method, "field": field, **metrics})
        for name, mask in regions.items():
            region_zero, region_ridge = _paired_metrics(truth, prediction, mask)
            for method, metrics in (
                ("zero_fluctuation", region_zero),
                ("ridge", region_ridge),
            ):
                region_records.append(
                    {
                        "method": method,
                        "field": field,
                        "region": name,
                        **metrics,
                    }
                )
        for bin_index, (lower, upper) in enumerate(
            zip(DISTANCE_BIN_EDGES_M[:-1], DISTANCE_BIN_EDGES_M[1:])
        ):
            distance_mask = (distance_m >= lower) & (distance_m < upper)
            distance_mask &= strict_xy[..., None]
            if not np.any(distance_mask):
                continue
            distance_zero, distance_ridge = _paired_metrics(
                truth, prediction, distance_mask
            )
            for method, metrics in (
                ("zero_fluctuation", distance_zero),
                ("ridge", distance_ridge),
            ):
                distance_records.append(
                    {
                        "method": method,
                        "field": field,
                        "distance_bin": bin_index,
                        "lower_m": lower,
                        "upper_m": "inf" if math.isinf(upper) else upper,
                        **metrics,
                    }
                )
        for method, estimate in (
            ("zero_fluctuation", np.zeros_like(truth)),
            ("ridge", prediction),
        ):
            for record in toroidal_mode_metrics(truth, estimate, strict_xy):
                mode_records.append({"method": method, "field": field, **record})
    return (
        field_records,
        region_records,
        distance_records,
        mode_records,
        hero_truth,
        hero_prediction,
    )


def _footprint_payload(footprints: Sequence[Any], omitted: Sequence[Any]) -> dict[str, Any]:
    def record(item: Any) -> dict[str, Any]:
        return {
            "family": item.family,
            "channel": item.channel,
            "center_xyz": list(item.center),
            "retained_cells": item.retained_cells,
            "nominal_cells": item.nominal_cells,
            "retained_fraction": item.retained_fraction,
        }

    return {
        "description": "BES/GPI-like localized density averages; not faithful diagnostics",
        "boxcar_shape_xyz": [3, 3, 5],
        "toroidal_wrap_only": True,
        "kept": [record(item) for item in footprints],
        "omitted": [record(item) for item in omitted],
        "preregistered_centers": {
            key: [list(center) for center in value]
            for key, value in FOOTPRINT_CENTERS.items()
        },
    }


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.geometry, args.geometry_manifest, args.paper0_root):
        assert_development_path(path)
    output = args.output.resolve()
    assert_development_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite S0 output {output}")
    output.mkdir(parents=True)

    actual_commit = _git(args.paper0_root, "rev-parse", "HEAD")
    if actual_commit != args.paper0_commit:
        raise ValueError("paper0 commit differs from requested revision")
    dirty = _git(args.paper0_root, "status", "--porcelain", "--untracked-files=all")
    source = OfficialS0Source(args.artifact_root)
    try:
        geometry, radius, vertical, geometry_manifest = _load_geometry(
            args.geometry, args.geometry_manifest
        )
        region_masks = geometry.region_masks
        strict_xy = region_masks.strict_wall_interior & region_masks.operator_interior
        footprints, omitted = build_fixed_footprints(strict_xy)
        grouped = group_footprints(footprints)
        observed = grouped["A"] + grouped["B"]
        heldout = grouped["C"]

        climatology = _training_climatology(source)
        train_frames = _frame_range(TRAIN_INTERVAL)
        validation_frames = _frame_range(VALIDATION_INTERVAL)
        train_observed = _observations(
            source, climatology[0], train_frames, observed
        )
        train_heldout = _observations(
            source, climatology[0], train_frames, heldout
        )
        validation_observed = _observations(
            source, climatology[0], validation_frames, observed
        )
        validation_heldout = _observations(
            source, climatology[0], validation_frames, heldout
        )

        fit_slice = slice(*INTERNAL_FIT_INTERVAL)
        tune_slice = slice(
            INTERNAL_TUNE_INTERVAL[0] - TRAIN_INTERVAL[0],
            INTERNAL_TUNE_INTERVAL[1] - TRAIN_INTERVAL[0],
        )
        selected, selection_records = _fit_and_tune(
            source,
            climatology,
            train_observed[fit_slice],
            train_observed[tune_slice],
            train_heldout[fit_slice],
            train_heldout[tune_slice],
            strict_xy,
        )
        final_kernel = DualRidgeKernel.fit(train_observed, selected)
        validation_c_prediction = final_kernel.predict_equivalent_dual(
            validation_observed, train_heldout
        )
        c_records = [
            {
                "method": "zero_fluctuation",
                **basic_metrics(validation_heldout, np.zeros_like(validation_heldout)),
            },
            {
                "method": "ridge",
                **basic_metrics(validation_heldout, validation_c_prediction),
            },
        ]
        zero_c_mse = float(c_records[0]["rmse"]) ** 2
        c_records[0]["relative_mse_skill_vs_zero"] = 0.0
        c_records[1]["relative_mse_skill_vs_zero"] = (
            1.0 - float(c_records[1]["rmse"]) ** 2 / zero_c_mse
            if zero_c_mse > 0.0
            else 0.0
        )
        hero_frame, per_frame_nrmse = select_median_hero_frame(
            validation_frames, validation_heldout, validation_c_prediction
        )
        distance_m = minimum_cylindrical_distance_to_observations(
            radius, vertical, observed
        )
        regions = {
            "outboard_midplane": region_masks.outboard_midplane,
            "x_point_stencil": region_masks.x_point_topology_stencil,
            "confined_edge": region_masks.confined_edge,
            "private_flux": region_masks.private_flux,
            "scrape_off_layer": region_masks.scrape_off_layer,
        }
        (
            field_records,
            region_records,
            distance_records,
            mode_records,
            hero_truth,
            hero_prediction,
        ) = _evaluate_full_state(
            source,
            climatology,
            final_kernel,
            train_observed,
            validation_observed,
            strict_xy,
            regions,
            distance_m,
            hero_frame,
        )
    finally:
        source.close()

    footprint_payload = _footprint_payload(footprints, omitted)
    write_strict_json_atomic(output / "footprints.json", footprint_payload)
    _write_csv(output / "ridge_selection.csv", selection_records)
    _write_csv(output / "heldout_c_metrics.csv", c_records)
    _write_csv(output / "full_state_metrics.csv", field_records)
    _write_csv(output / "region_skill.csv", region_records)
    _write_csv(output / "distance_skill.csv", distance_records)
    _write_csv(output / "mode_skill.csv", mode_records)
    mode_band_records: list[dict[str, Any]] = []
    for method in ("zero_fluctuation", "ridge"):
        for field in S0_FIELDS:
            selected_modes = [
                row
                for row in mode_records
                if row["method"] == method and row["field"] == field
            ]
            for band, predicate in (
                ("low_n_5_to_15", lambda n: 5 <= n <= 15),
                ("evaluated_n_20_to_35", lambda n: 20 <= n <= 35),
                ("higher_n_ge_40", lambda n: n >= 40),
            ):
                rows = [row for row in selected_modes if predicate(int(row["physical_n"]))]
                truth_power = float(sum(float(row["truth_power"]) for row in rows))
                prediction_power = float(
                    sum(float(row["prediction_power"]) for row in rows)
                )
                mode_band_records.append(
                    {
                        "method": method,
                        "field": field,
                        "band": band,
                        "mode_count": len(rows),
                        "truth_power": truth_power,
                        "prediction_power": prediction_power,
                        "retained_power_ratio": (
                            prediction_power / truth_power if truth_power > 0.0 else 0.0
                        ),
                        "mean_coefficient_mse": float(
                            np.mean([float(row["coefficient_mse"]) for row in rows])
                        ),
                    }
                )
    _write_csv(output / "mode_band_skill.csv", mode_band_records)
    _write_csv(
        output / "hero_selection.csv",
        [
            {
                "frame": frame,
                "heldout_c_nrmse": float(score),
                "selected": frame == hero_frame,
            }
            for frame, score in zip(validation_frames, per_frame_nrmse)
        ],
    )
    footprint_labels = np.zeros(VOLUME_SHAPE, dtype=np.int8)
    for family, label in (("A", 1), ("B", 2), ("C", 3)):
        for footprint in grouped[family]:
            footprint_labels.ravel()[footprint.flat_indices] = label
    np.savez_compressed(
        output / "hero_frame.npz",
        frame=np.int64(hero_frame),
        fields=np.asarray(S0_FIELDS),
        truth=hero_truth,
        ridge=hero_prediction,
        error=hero_prediction - hero_truth,
        observed_ab=validation_observed[hero_frame - VALIDATION_INTERVAL[0]],
        heldout_c_truth=validation_heldout[hero_frame - VALIDATION_INTERVAL[0]],
        heldout_c_prediction=validation_c_prediction[
            hero_frame - VALIDATION_INTERVAL[0]
        ],
        major_radius_m=radius,
        vertical_position_m=vertical,
        strict_operator_mask=strict_xy,
        minimum_distance_to_ab_m=distance_m,
        footprint_labels=footprint_labels,
    )
    np.savez_compressed(
        output / "validation_diagnostics.npz",
        frames=np.asarray(validation_frames, dtype=np.int64),
        observed_ab=validation_observed,
        heldout_c_truth=validation_heldout,
        heldout_c_ridge=validation_c_prediction,
    )

    result = {
        "schema_version": 1,
        "status": "s0_spatial_linear_reconstruction_complete",
        "development_run": "85604",
        "training_performed": True,
        "training_kind": "closed_form_regularized_linear_reconstruction_only",
        "temporal_forecasting_performed": False,
        "protocol": PROTOCOL,
        "splits": {
            "train": list(TRAIN_INTERVAL),
            "internal_fit": list(INTERNAL_FIT_INTERVAL),
            "internal_guard": list(INTERNAL_GUARD_INTERVAL),
            "internal_tune": list(INTERNAL_TUNE_INTERVAL),
            "official_guard": [432, 496],
            "validation": list(VALIDATION_INTERVAL),
        },
        "fields": list(S0_FIELDS),
        "normalization": "official train-only scalar normalization then per-cell train climatology subtraction",
        "selected_regularization": selected,
        "regularization_grid": list(RIDGE_LAMBDAS),
        "ridge_output_x_slab": OUTPUT_X_SLAB,
        "strict_operator_cell_count": int(np.sum(strict_xy)),
        "diagnostic_channel_counts": {
            family: len(grouped[family]) for family in ("A", "B", "C")
        },
        "omitted_channel_count": len(omitted),
        "hero_frame": hero_frame,
        "heldout_c": {record["method"]: record for record in c_records},
        "paper0_commit": actual_commit,
        "dirty_at_execution": bool(dirty),
        "slurm_job_id": str(args.slurm_job_id),
    }
    write_strict_json_atomic(output / "result.json", result)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "paper0_commit": actual_commit,
        "dirty_state": dirty.splitlines(),
        "seed": None,
        "inputs": {
            "artifact_root": str(args.artifact_root),
            "model_shards": [
                {
                    "path": str(shard.path),
                    "sha256": shard.sha256,
                    "frames": [shard.start, shard.stop],
                }
                for shard in source.catalog.shards
            ],
            "geometry_path": str(args.geometry),
            "geometry_sha256": geometry_manifest["sources"]["geometry"]["sha256"],
            "geometry_manifest": str(args.geometry_manifest),
            "geometry_manifest_sha256": sha256_file(args.geometry_manifest),
        },
        "outputs": sorted(path.name for path in output.iterdir()),
        "selected_frame": hero_frame,
        "horizons": [],
        "scope": "simultaneous_spatial_reconstruction_no_temporal_forecast",
    }
    write_strict_json_atomic(output / "run_manifest.json", manifest)
    tracked = sorted(
        path
        for path in output.iterdir()
        if path.name != "artifact_sha256.txt" and path.is_file()
    )
    with (output / "artifact_sha256.txt").open("x", encoding="utf-8") as handle:
        for path in tracked:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
