#!/usr/bin/env python3
"""Build the frozen old-85604, seed-1702 physics-first forecast figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from tcv_diagnostics.b5_covariance_localization import (
    exact_separatrix_local_contributions,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.codec_transport import (
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from tcv_diagnostics.matched_o1_transport import (
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
)
from tcv_diagnostics.o2_context_data import OneStepContextDataset
from tcv_diagnostics.o2_training_data import OneStepWindowDataset
from tcv_diagnostics.physics_first_figures import (
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MODEL_LABELS,
    NePhiCrossSpectrumAccumulator,
    OneStepForecastReader,
    PHYSICS_FIGURE_H1_TARGETS,
    PHYSICS_FIGURE_MEMBERS,
    PHYSICS_FIGURE_MODEL_SEED,
    PHYSICS_FIGURE_STARTS,
    PHYSICS_FIGURE_TARGET,
    PHYSICS_FIGURE_Z_PLANE,
    PersistentForecastReader,
    TRANSPORT_VARIOGRAM_LAGS,
    authoritative_local_transport,
    bootstrap_curve_mean,
    build_plot_geometry,
    decode_c5p_members,
    first_order_toroidal_variogram,
    lower_sample_median_target,
    moving_block_bootstrap_indices,
    save_field_comparison_figure,
    save_ne_phi_coupling_figure,
    save_transport_profile_figure,
    save_transport_variogram_figure,
    standardized_toroidal_fluctuations,
)


DEFAULT_MODEL_DATA = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
)
DEFAULT_CONTEXT_FORECAST = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_scientific_evaluation_full/"
    "job_6913512/task_4_b5_context_seed_1702/evaluation/forecast_M32.h5"
)
DEFAULT_CONTEXT_SCORE = DEFAULT_CONTEXT_FORECAST.with_name("score.json")
DEFAULT_ECRD_FORECAST = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_scientific_evaluation_full/"
    "job_6913512/task_7_ecrd_seed_1702/evaluation/forecast_M32.h5"
)
DEFAULT_PERSISTENT_FORECAST = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/"
    "post_ecrd_old_85604_persistent_global_local_physics_evaluation/"
    "job_6938347/generation/forecast_M32_four_frame.h5"
)
DEFAULT_GEOMETRY = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
)
EXPECTED_CONTEXT_FORECAST_SHA = (
    "303d0c000b86bb3a90356469ab03f31b4080b6d4b5af5d8d0b538788bef3ad4e"
)
EXPECTED_CONTEXT_SCORE_SHA = (
    "52c2d961442528f034f668b0305eed4b3b080d6c8b957eecf0fa14f01d4baf40"
)
EXPECTED_ECRD_FORECAST_SHA = (
    "1212dd384c5beb10cf2c46f799708402868cf60f228c980b7f991c556d6e960e"
)
EXPECTED_PERSISTENT_FORECAST_SHA = (
    "60a6926dde5c765081b7ac578966036c0769aff8c538317265e93673da10f4ae"
)
EXPECTED_GEOMETRY_SHA = (
    "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
)
EXPECTED_GEOMETRY_MANIFEST_SHA = (
    "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
)
EXPECTED_NATIVE_TRUTH_RESULT_SHA = (
    "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_MODEL_DATA)
    parser.add_argument("--context-forecast", type=Path, default=DEFAULT_CONTEXT_FORECAST)
    parser.add_argument("--context-score", type=Path, default=DEFAULT_CONTEXT_SCORE)
    parser.add_argument("--ecrd-forecast", type=Path, default=DEFAULT_ECRD_FORECAST)
    parser.add_argument("--persistent-forecast", type=Path, default=DEFAULT_PERSISTENT_FORECAST)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument(
        "--geometry-manifest",
        type=Path,
        default=Path("paper0/manifests/phase2_85604_geometry_units.json"),
    )
    parser.add_argument(
        "--native-truth-result",
        type=Path,
        default=Path("paper0/results/phase2_potential_vorticity_all_frame_6893033.json"),
    )
    parser.add_argument(
        "--deterministic-result",
        type=Path,
        default=Path(
            "paper0/results/post_ecrd_old_85604_stage2_multilead_seed1702_6936642.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_checkout(root: Path, expected: str) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != expected or dirty:
        raise RuntimeError("physics-figure checkout is not the locked clean commit")
    return {"commit": head, "dirty": False}


def _verify_file(path: Path, expected: str, *, name: str) -> dict[str, Any]:
    assert_development_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_path(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 differs: {actual} != {expected}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def _load_deterministic_model(
    result_path: Path, *, device: torch.device
) -> tuple[CodecFreeIncrementOperator3D, dict[str, Any]]:
    result = load_strict_json(result_path)
    if (
        result.get("scope") != "post_ecrd_old_85604_stage2_multilead_scaling"
        or result.get("development_run") != "85604"
        or int(result.get("seed", -1)) != PHYSICS_FIGURE_MODEL_SEED
        or result.get("held_out_85606_read") is not False
        or result.get("guard_frames_read") is not False
        or result.get("prospective_gate_passed") is not True
    ):
        raise ValueError("deterministic multilead result contract differs")
    selected = result["best_checkpoint"]
    if int(selected["epoch"]) != 4:
        raise ValueError("deterministic selected epoch differs")
    checkpoint = Path(selected["path"])
    checkpoint_record = _verify_file(
        checkpoint, str(selected["sha256"]), name="deterministic checkpoint"
    )
    architecture = result["architecture"]["architecture"]
    config = CodecFreeOperatorConfig(
        state_family="c5p",
        history_frames=1,
        base_channels=int(architecture["base_channels"]),
        channel_multipliers=tuple(architecture["channel_multipliers"]),
        blocks_per_level=int(architecture["blocks_per_level"]),
        lead_embedding_channels=int(architecture["lead_embedding_channels"]),
        group_norm_maximum_groups=int(architecture["group_norm_maximum_groups"]),
        kernel_size=int(architecture["kernel_size"]),
        predict_boundary=False,
        zero_initialize_output=bool(architecture["zero_initialize_output"]),
        auxiliary_context_channels=int(architecture["auxiliary_context_channels"]),
    )
    if config.to_record() != architecture:
        raise ValueError("deterministic serialized architecture differs")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if (
        payload.get("family") != "c5p"
        or payload.get("stage") != "stage2_multilead_finetune"
        or int(payload.get("seed", -1)) != PHYSICS_FIGURE_MODEL_SEED
        or int(payload.get("epoch", -1)) != 4
        or payload.get("config") != architecture
    ):
        raise ValueError("deterministic checkpoint payload differs")
    model = CodecFreeIncrementOperator3D(config).to(device, torch.float32)
    model.load_state_dict(payload["model"], strict=True)
    if not all(
        torch.equal(value.to(device), model.state_dict()[name])
        for name, value in payload["model"].items()
    ):
        raise AssertionError("deterministic checkpoint did not reload bitwise")
    model.eval()
    return model, {
        "result": {"path": str(result_path.resolve()), "sha256": sha256_path(result_path)},
        "checkpoint": {**checkpoint_record, "epoch": 4},
        "architecture": result["architecture"],
        "checkpoint_reload_bitwise": True,
    }


def _native_local_truth(native: Mapping[str, np.ndarray], geometry: Any) -> tuple[dict[str, np.ndarray], float]:
    evaluated = evaluate_transport_state(
        direct_pressure_transport_state(
            native["Ne"], native["Pe"], native["Pi"], native["phi"]
        ),
        geometry,
    )
    return exact_separatrix_local_contributions(
        evaluated,
        strict_face_mask=geometry.strict_face_mask,
        separatrix_face_mask=geometry.separatrix_face_mask,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cross_rows(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, record in records.items():
        for index, k in enumerate(np.asarray(record["stored_k"], dtype=np.int64)):
            rows.append(
                {
                    "model_key": key,
                    "model_label": "truth" if key == "truth" else MODEL_LABELS[key],
                    "horizon_frames": 1,
                    "stored_k": int(k),
                    "physical_n": int(np.asarray(record["physical_n"])[index]),
                    "cross_magnitude": float(np.asarray(record["cross_magnitude"])[index]),
                    "phase_degrees": float(np.asarray(record["phase_degrees"])[index]),
                    "coherence": float(np.asarray(record["coherence"])[index]),
                    "sample_count": int(record["sample_count"]),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    root = args.paper0_root.resolve(strict=True)
    for attr in ("geometry_manifest", "native_truth_result", "deterministic_result"):
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, root / value)
    for path in (
        args.artifact_root,
        args.context_forecast,
        args.context_score,
        args.ecrd_forecast,
        args.persistent_forecast,
        args.geometry,
        args.geometry_manifest,
        args.native_truth_result,
        args.deterministic_result,
        args.output,
    ):
        assert_development_path(path)
    checkout = _verify_checkout(root, args.paper0_commit)
    if args.output.exists():
        raise FileExistsError(args.output)
    partial = args.output.with_name(f".{args.output.name}.partial")
    if partial.exists():
        raise FileExistsError(partial)

    sources = {
        "conditioned_forecast": _verify_file(
            args.context_forecast,
            EXPECTED_CONTEXT_FORECAST_SHA,
            name="conditioned forecast",
        ),
        "conditioned_score": _verify_file(
            args.context_score, EXPECTED_CONTEXT_SCORE_SHA, name="conditioned score"
        ),
        "ecrd_forecast": _verify_file(
            args.ecrd_forecast, EXPECTED_ECRD_FORECAST_SHA, name="ECRD forecast"
        ),
        "persistent_forecast": _verify_file(
            args.persistent_forecast,
            EXPECTED_PERSISTENT_FORECAST_SHA,
            name="persistent forecast",
        ),
        "geometry": _verify_file(
            args.geometry, EXPECTED_GEOMETRY_SHA, name="transport geometry"
        ),
        "geometry_manifest": _verify_file(
            args.geometry_manifest,
            EXPECTED_GEOMETRY_MANIFEST_SHA,
            name="geometry manifest",
        ),
        "native_truth_result": _verify_file(
            args.native_truth_result,
            EXPECTED_NATIVE_TRUTH_RESULT_SHA,
            name="native truth result",
        ),
    }
    score = load_strict_json(args.context_score)
    per_target_score = score["memberwise_transport"]["overall"]["per_target"]
    selected_target, ordered_selection = lower_sample_median_target(per_target_score)
    if selected_target != PHYSICS_FIGURE_TARGET:
        raise RuntimeError(
            f"conditioned-score representative is {selected_target}, not frozen 509"
        )
    expected_selected_error = float(
        next(
            item["absolute_error"]
            for item in ordered_selection
            if int(item["target_frame"]) == selected_target
        )
    )

    geometry_manifest = load_strict_json(args.geometry_manifest)
    geometry = load_transport_geometry(
        geometry_path=args.geometry, geometry_manifest=geometry_manifest
    )
    masks = geometry.region_masks
    eligible_xy = np.asarray(
        masks.strict_wall_interior & masks.operator_interior, dtype=bool
    )
    grid = geometry_manifest["grid"]
    offset = int(grid["model_x_to_grid_x_offset"])
    ixseps = int(grid["topology"]["ixseps1"])
    core_start, core_stop_inclusive = map(
        int, grid["topology"]["core_y_inclusive"]
    )
    with h5py.File(args.geometry, "r") as handle:
        rxy = np.asarray(handle["Rxy"][offset : offset + 64], dtype=np.float64)
        zxy = np.asarray(handle["Zxy"][offset : offset + 64], dtype=np.float64)
        face_r = np.asarray(
            handle["Rxy_xlow"][ixseps, core_start : core_stop_inclusive + 1],
            dtype=np.float64,
        )
        face_z = np.asarray(
            handle["Zxy_xlow"][ixseps, core_start : core_stop_inclusive + 1],
            dtype=np.float64,
        )
    plot_geometry = build_plot_geometry(
        major_radius=rxy,
        vertical_position=zxy,
        separatrix_major_radius=face_r,
        separatrix_vertical_position=face_z,
        native_dz=geometry.dz,
    )

    catalog = load_official_catalog(args.artifact_root)
    native_catalog = NativeTruthCatalog(load_strict_json(args.native_truth_result))
    all_validation = tuple(range(498, 624))
    truth_dataset = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=all_validation,
        context_frames=1,
        augment=False,
        seed=PHYSICS_FIGURE_MODEL_SEED,
    )
    truth_cross = NePhiCrossSpectrumAccumulator(eligible_xy)
    h1_target_set = set(PHYSICS_FIGURE_H1_TARGETS)
    h4_targets = tuple(start + 4 for start in PHYSICS_FIGURE_STARTS)
    union_targets = tuple(sorted(h1_target_set | set(h4_targets)))
    truth_local: dict[int, np.ndarray] = {}
    truth_std_target: np.ndarray | None = None
    truth_variogram_h1: list[np.ndarray] = []
    truth_variogram_h4: list[np.ndarray] = []
    maximum_closure = 0.0
    try:
        for target in union_targets:
            item = truth_dataset[target - 498]
            standardized = np.asarray(item["target"], dtype=np.float32)
            if int(item["target_frame_index"]) != target:
                raise ValueError("truth target coordinate differs")
            if target in h1_target_set:
                truth_cross.update(standardized[None])
            if target == PHYSICS_FIGURE_TARGET:
                truth_std_target = standardized
            native = native_catalog.read(target, target + 1)
            local, closure = _native_local_truth(native, geometry)
            maximum_closure = max(maximum_closure, closure)
            particle = np.asarray(local["particle"][0], dtype=np.float64)
            truth_local[target] = particle
            if target in h1_target_set:
                truth_variogram_h1.append(first_order_toroidal_variogram(particle[None])[0])
            if target in set(h4_targets):
                truth_variogram_h4.append(first_order_toroidal_variogram(particle[None])[0])
    finally:
        truth_dataset.close()
    if truth_std_target is None:
        raise RuntimeError("representative truth field was not loaded")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("deterministic figure inference requires allocated CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, deterministic_provenance = _load_deterministic_model(
        args.deterministic_result, device=device
    )
    sources["deterministic_model"] = deterministic_provenance
    context_dataset = OneStepContextDataset(
        catalog,
        target_frames=tuple(range(498, 621)),
        context_frames=1,
    )
    deterministic_cross = NePhiCrossSpectrumAccumulator(eligible_xy)
    deterministic_variogram_h1: list[np.ndarray] = []
    field_means: dict[str, np.ndarray] = {}
    target_transport: dict[str, np.ndarray] = {}
    try:
        with torch.inference_mode():
            for target in PHYSICS_FIGURE_H1_TARGETS:
                item = context_dataset[target - 498]
                if int(item["target_frame_index"]) != target or item["target_truth_read"] is not False:
                    raise ValueError("deterministic context contract differs")
                context = torch.from_numpy(item["context"]).unsqueeze(0).to(device)
                lead = torch.ones(1, dtype=torch.float32, device=device)
                prediction = model.forecast(context, lead).volume[0]
                standardized = prediction.detach().cpu().numpy().astype(np.float32)
                deterministic_cross.update(standardized[None])
                physical = decode_c5p_members(catalog, standardized[None])
                native = native_catalog.read(target, target + 1)
                local, checked_truth, closure = authoritative_local_transport(
                    physical_model88=physical,
                    native_truth=native,
                    geometry=geometry,
                )
                maximum_closure = max(maximum_closure, closure)
                if not np.allclose(
                    checked_truth["particle"][0], truth_local[target], rtol=0.0, atol=0.0
                ):
                    raise RuntimeError("deterministic transport truth replay differs")
                deterministic_variogram_h1.append(
                    first_order_toroidal_variogram(local["particle"])[0]
                )
                if target == PHYSICS_FIGURE_TARGET:
                    field_means["deterministic"] = standardized
                    target_transport["deterministic"] = local["particle"]
    finally:
        context_dataset.close()
        del model
        torch.cuda.empty_cache()

    cross_accumulators = {
        "b5_context": NePhiCrossSpectrumAccumulator(eligible_xy),
        "ecrd": NePhiCrossSpectrumAccumulator(eligible_xy),
        "persistent": NePhiCrossSpectrumAccumulator(eligible_xy),
    }
    variogram_per_target = {
        "b5_context": [],
        "ecrd": [],
        "persistent": [],
    }

    def consume_h1(key: str, members: np.ndarray, target: int) -> None:
        nonlocal maximum_closure
        cross_accumulators[key].update(members)
        physical = decode_c5p_members(catalog, members)
        native = native_catalog.read(target, target + 1)
        local, checked_truth, closure = authoritative_local_transport(
            physical_model88=physical,
            native_truth=native,
            geometry=geometry,
        )
        maximum_closure = max(maximum_closure, closure)
        if not np.allclose(
            checked_truth["particle"][0], truth_local[target], rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"{key} transport truth replay differs")
        member_curves = first_order_toroidal_variogram(local["particle"])
        variogram_per_target[key].append(np.mean(member_curves, axis=0))
        if target == PHYSICS_FIGURE_TARGET:
            field_means[key] = np.mean(members, axis=0, dtype=np.float64).astype(np.float32)
            target_transport[key] = local["particle"]

    with OneStepForecastReader(
        args.context_forecast, arm="B5-Context", model_seed=1702
    ) as reader:
        for target in PHYSICS_FIGURE_H1_TARGETS:
            consume_h1("b5_context", reader.read(target), target)
    with OneStepForecastReader(
        args.ecrd_forecast, arm="ECRD", model_seed=1702
    ) as reader:
        for target in PHYSICS_FIGURE_H1_TARGETS:
            consume_h1("ecrd", reader.read(target), target)
    pgl_h4_per_target: list[np.ndarray] = []
    with PersistentForecastReader(args.persistent_forecast) as reader:
        for start in PHYSICS_FIGURE_STARTS:
            consume_h1("persistent", reader.read(start=start, horizon=1), start + 1)
        for start in PHYSICS_FIGURE_STARTS:
            target = start + 4
            members = reader.read(start=start, horizon=4)
            physical = decode_c5p_members(catalog, members)
            native = native_catalog.read(target, target + 1)
            local, checked_truth, closure = authoritative_local_transport(
                physical_model88=physical,
                native_truth=native,
                geometry=geometry,
            )
            maximum_closure = max(maximum_closure, closure)
            if not np.allclose(
                checked_truth["particle"][0], truth_local[target], rtol=0.0, atol=0.0
            ):
                raise RuntimeError("persistent h4 transport truth replay differs")
            pgl_h4_per_target.append(
                np.mean(first_order_toroidal_variogram(local["particle"]), axis=0)
            )

    if tuple(field_means) != tuple(MODEL_LABELS) or tuple(target_transport) != tuple(MODEL_LABELS):
        raise RuntimeError("representative cross-model outputs are incomplete")
    b5_wedge_error = abs(
        float(np.sum(np.mean(target_transport["b5_context"], axis=0)))
        - float(np.sum(truth_local[PHYSICS_FIGURE_TARGET]))
    )
    # The stored score reconstructs absolute error from JSON-serialized summary
    # factors, whereas this path repeats the float32 88->81 resampling and
    # nonlinear transport evaluation.  Require agreement to one part per
    # million; bit-level agreement is neither available nor scientifically
    # meaningful across those two reduction paths.
    if not np.isclose(b5_wedge_error, expected_selected_error, rtol=1.0e-6, atol=1.0e-9):
        raise RuntimeError(
            "recomputed conditioned target-selection transport error differs "
            f"({b5_wedge_error} != {expected_selected_error})"
        )

    cross_records = {
        "truth": truth_cross.finalize(),
        "deterministic": deterministic_cross.finalize(),
        **{key: accumulator.finalize() for key, accumulator in cross_accumulators.items()},
    }
    bootstrap_indices = moving_block_bootstrap_indices(len(PHYSICS_FIGURE_STARTS))
    h1_intervals = {
        "truth": bootstrap_curve_mean(np.asarray(truth_variogram_h1), bootstrap_indices),
        "deterministic": bootstrap_curve_mean(
            np.asarray(deterministic_variogram_h1), bootstrap_indices
        ),
        **{
            key: bootstrap_curve_mean(np.asarray(values), bootstrap_indices)
            for key, values in variogram_per_target.items()
        },
    }
    h4_intervals = {
        "truth": bootstrap_curve_mean(np.asarray(truth_variogram_h4), bootstrap_indices),
        "persistent": bootstrap_curve_mean(
            np.asarray(pgl_h4_per_target), bootstrap_indices
        ),
    }

    partial.mkdir(parents=True)
    figures = partial / "figures"
    tables = partial / "tables"
    figures.mkdir()
    tables.mkdir()
    save_field_comparison_figure(
        figures / "matched_one_step_fields.png",
        truth=truth_std_target,
        forecast_means=field_means,
        eligible_xy=eligible_xy,
        plot_geometry=plot_geometry,
        region_masks=masks,
    )
    save_transport_profile_figure(
        figures / "memberwise_particle_transport.png",
        truth_local=truth_local[PHYSICS_FIGURE_TARGET],
        forecast_local=target_transport,
        poloidal_distance=plot_geometry.separatrix_arc_length,
    )
    save_transport_variogram_figure(
        figures / "particle_transport_variogram.png",
        separation_m=plot_geometry.toroidal_separations,
        h1=h1_intervals,
        pgl_h4=h4_intervals,
    )
    save_ne_phi_coupling_figure(
        figures / "ne_phi_coupling.png", records=cross_records
    )

    field_npz: dict[str, np.ndarray] = {
        "R_m": plot_geometry.major_radius,
        "Z_m": plot_geometry.vertical_position,
        "eligible_xy": eligible_xy,
        "truth_Ne_z44": standardized_toroidal_fluctuations(
            truth_std_target[None], eligible_xy=eligible_xy
        )[0, 0, :, :, PHYSICS_FIGURE_Z_PLANE],
        "truth_phi_z44": standardized_toroidal_fluctuations(
            truth_std_target[None], eligible_xy=eligible_xy
        )[0, 3, :, :, PHYSICS_FIGURE_Z_PLANE],
    }
    for key, values in field_means.items():
        prepared = standardized_toroidal_fluctuations(
            values[None], eligible_xy=eligible_xy
        )[0]
        field_npz[f"{key}_Ne_z44"] = prepared[0, :, :, PHYSICS_FIGURE_Z_PLANE]
        field_npz[f"{key}_phi_z44"] = prepared[3, :, :, PHYSICS_FIGURE_Z_PLANE]
    np.savez_compressed(tables / "matched_fields_target509.npz", **field_npz)

    transport_rows: list[dict[str, Any]] = []
    truth_curve = np.sum(truth_local[PHYSICS_FIGURE_TARGET], axis=-1)
    for row, (distance, value) in enumerate(
        zip(plot_geometry.separatrix_arc_length, truth_curve)
    ):
        transport_rows.append(
            {
                "model_key": "truth",
                "model_label": "truth",
                "member": -1,
                "poloidal_row": core_start + row,
                "s_m": float(distance),
                "particle_contribution": float(value),
                "cumulative_particle_transport": float(np.sum(truth_curve[: row + 1])),
            }
        )
    for key, local in target_transport.items():
        curves = np.sum(local, axis=-1)
        for member, curve in enumerate(curves):
            cumulative = np.cumsum(curve)
            for row, distance in enumerate(plot_geometry.separatrix_arc_length):
                transport_rows.append(
                    {
                        "model_key": key,
                        "model_label": MODEL_LABELS[key],
                        "member": member,
                        "poloidal_row": core_start + row,
                        "s_m": float(distance),
                        "particle_contribution": float(curve[row]),
                        "cumulative_particle_transport": float(cumulative[row]),
                    }
                )
    _write_csv(
        tables / "memberwise_particle_transport_target509.csv",
        [
            "model_key",
            "model_label",
            "member",
            "poloidal_row",
            "s_m",
            "particle_contribution",
            "cumulative_particle_transport",
        ],
        transport_rows,
    )

    variogram_rows: list[dict[str, Any]] = []
    for horizon, records in ((1, h1_intervals), (4, h4_intervals)):
        for key, record in records.items():
            for index, lag in enumerate(TRANSPORT_VARIOGRAM_LAGS):
                variogram_rows.append(
                    {
                        "model_key": key,
                        "model_label": "truth" if key == "truth" else MODEL_LABELS[key],
                        "horizon_frames": horizon,
                        "horizon_microseconds": horizon * 3.1319,
                        "lag_native_cells": lag,
                        "mean_toroidal_arc_separation_m": float(plot_geometry.toroidal_separations[index]),
                        "mean": float(record["mean"][index]),
                        "lower_2p5": float(record["lower_2p5"][index]),
                        "upper_97p5": float(record["upper_97p5"][index]),
                    }
                )
    _write_csv(
        tables / "particle_transport_variogram.csv",
        [
            "model_key",
            "model_label",
            "horizon_frames",
            "horizon_microseconds",
            "lag_native_cells",
            "mean_toroidal_arc_separation_m",
            "mean",
            "lower_2p5",
            "upper_97p5",
        ],
        variogram_rows,
    )
    _write_csv(
        tables / "ne_phi_coupling.csv",
        [
            "model_key",
            "model_label",
            "horizon_frames",
            "stored_k",
            "physical_n",
            "cross_magnitude",
            "phase_degrees",
            "coherence",
            "sample_count",
        ],
        _cross_rows(cross_records),
    )

    result = {
        "schema_version": 1,
        "scope": "old_85604_physics_first_matched_forecast_figures",
        "status": "completed",
        "development_run": "85604",
        "model_seed": 1702,
        "shared_horizon_frames": 1,
        "shared_horizon_microseconds": 3.1319,
        "shared_target_count": 36,
        "representative_target_frame": PHYSICS_FIGURE_TARGET,
        "representative_target_selection": {
            "model": MODEL_LABELS["b5_context"],
            "statistic": "lower_sample_median_absolute_error_of_ensemble_expectation_of_memberwise_particle_wedge_transport",
            "rank_one_indexed": 18,
            "population": 36,
            "stored_score_absolute_error": expected_selected_error,
            "recomputed_absolute_error": b5_wedge_error,
            "upper_central_target_descriptive_only": int(ordered_selection[18]["target_frame"]),
        },
        "field_plane": {"stored_z": 44, "physical_time_microseconds": 3.1319},
        "transport": {
            "quantity": "particle",
            "surface": "confined_separatrix",
            "model_face_left_x": 15,
            "poloidal_rows_inclusive": [8, 23],
            "native_toroidal_cells": 81,
            "nonlinear_operator_applied_memberwise": True,
            "transport_of_ensemble_mean_fields_used": False,
            "maximum_relative_sum_closure_error": maximum_closure,
        },
        "variogram": {
            "order": 1,
            "lags_native_cells": list(TRANSPORT_VARIOGRAM_LAGS),
            "bootstrap": {
                "method": "noncircular_chronological_moving_block",
                "block_length_selected_starts": BOOTSTRAP_BLOCK_LENGTH,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
        },
        "cross_field": {
            "fields": ["Ne", "phi"],
            "mode_mapping": "n=5k",
            "shaded_physical_band_n": [20, 35],
            "ensemble_reduction": "memberwise_cross_spectra_then_expectation",
        },
        "outputs": {
            "figures": [
                "figures/matched_one_step_fields.png",
                "figures/memberwise_particle_transport.png",
                "figures/particle_transport_variogram.png",
                "figures/ne_phi_coupling.png",
            ],
            "tables": [
                "tables/matched_fields_target509.npz",
                "tables/memberwise_particle_transport_target509.csv",
                "tables/particle_transport_variogram.csv",
                "tables/ne_phi_coupling.csv",
            ],
        },
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "target_truth_used_during_deterministic_inference": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    _atomic_json(partial / "result.json", result)
    manifest = {
        "schema_version": 1,
        "scope": result["scope"],
        "command": [sys.executable, *sys.argv],
        "slurm_job_id": args.slurm_job_id,
        "checkout": checkout,
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "accelerator": torch.cuda.get_device_name(0),
        },
        "sources": sources,
        "frozen_population": {
            "starts": list(PHYSICS_FIGURE_STARTS),
            "h1_targets": list(PHYSICS_FIGURE_H1_TARGETS),
            "model_seed": 1702,
            "ensemble_members": PHYSICS_FIGURE_MEMBERS,
        },
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    _atomic_json(partial / "manifest.json", manifest)
    lines = []
    for path in sorted(partial.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256.txt":
            lines.append(f"{sha256_path(path)}  {path.relative_to(partial)}\n")
    (partial / "artifact_sha256.txt").write_text("".join(lines), encoding="utf-8")
    os.replace(partial, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
