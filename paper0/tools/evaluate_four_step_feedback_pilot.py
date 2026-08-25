#!/usr/bin/env python3
"""Compare the frozen four-step feedback pilot with its bitwise parent."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from paper0.tools.evaluate_codec_free_bounded_rollout import (
    HORIZONS,
    VALIDATION_START,
    VALIDATION_STOP,
    _create_delta_dataset,
    _frame_slice,
    _persistence_records,
    load_validation_states,
)
from paper0.tools.score_codec_free_bounded_rollout_physics import (
    SPEC,
    _json_safe,
    _native_truth_transport,
    _transport_slice,
    decode_batch,
    score_common_persistence,
    transport_from_model88,
)
from paper0.tools.train_codec_free_four_step_feedback import (
    authorize_manifest as authorize_training_manifest,
    verify_and_load_parent,
)
from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    verify_finished_wandb_run,
)
from paper0.tools.train_codec_free_stage2_multilead import (
    build_model,
    validate_parent_config,
)
from tcv_diagnostics.bounded_rollout import (
    FIELDS,
    FieldErrorAccumulator,
    autoregressive_forecast_path,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.codec_transport import TransportComparisonAccumulator
from tcv_diagnostics.matched_codec_metrics import MatchedCodecAccumulator
from tcv_diagnostics.matched_o1_transport import (
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import VOLUME_SHAPE, load_official_catalog
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SCOPE = "post_ecrd_old_85604_four_step_feedback_evaluation"
METHODS = ("pre_feedback_parent", "four_step_feedback_finetuned")
QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)
BANDS = ("k1_3", "k4_5", "k6_7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def repository_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_locked_file(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if "85606" in str(path).lower():
        raise ValueError("held-out 85606 paths are prohibited")
    if len(digest) != 64 or sha256_path(path) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return path


def authorize_evaluation_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> None:
    if sha256_path(manifest_path) != manifest_sha256:
        raise ValueError("four-step evaluation manifest SHA-256 differs")
    expected = {
        "scope": SCOPE,
        "status": "frozen_after_training_before_physics_evaluation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "steering_allowed": False,
        "physics_derived_loss_used": False,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "fields": list(FIELDS),
        "horizons": list(HORIZONS),
        "methods": list(METHODS),
        "inference_batch_size": 4,
        "wandb_required": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("four-step evaluation scope differs")
    gates = manifest.get("physics_preservation_gates", {})
    if gates != {
        "maximum_absolute_log_power_ratio_error_increase_fraction": 0.10,
        "maximum_mean_separatrix_relative_l2_increase_fraction": 0.05,
        "strict_face_transport_is_report_only": True,
        "cross_field_coherence_change_is_report_only": True,
    }:
        raise ValueError("four-step evaluation physics gates differ")


def authorize_training_result(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training_manifest_path = verify_locked_file(
        manifest.get("training_manifest", {}), label="training manifest"
    )
    training_manifest = load_strict_json(training_manifest_path)
    authorize_training_manifest(training_manifest, seed=1702)
    result_path = verify_locked_file(
        manifest.get("training_result", {}), label="training result"
    )
    result = load_strict_json(result_path)
    if (
        result.get("scope") != "post_ecrd_old_85604_four_step_feedback_pilot"
        or result.get("status") != "completed"
        or result.get("development_run") != "85604"
        or result.get("held_out_85606_read") is not False
        or result.get("new_nersc_data_read") is not False
        or result.get("guard_frames_read") is not False
        or result.get("training_performed") is not True
        or result.get("physics_derived_loss_used") is not False
        or result.get("physics_diagnostics_scored") is not False
        or result.get("physics_evaluation_authorized") is not True
        or result.get("mechanical_gate", {}).get("passed") is not True
        or int(result.get("seed", -1)) != 1702
        or result.get("manifest_sha256") != manifest["training_manifest"]["sha256"]
        or Path(str(result.get("manifest", ""))) != training_manifest_path
    ):
        raise ValueError("four-step training result does not authorize evaluation")
    checkpoint_path = verify_locked_file(
        result.get("best_checkpoint", {}), label="selected feedback checkpoint"
    )
    if checkpoint_path != Path(str(result["best_checkpoint"]["path"])):
        raise AssertionError("selected feedback checkpoint path vanished")
    return training_manifest, result, {
        "path": str(result_path),
        "sha256": manifest["training_result"]["sha256"],
    }


def load_models(
    training_manifest: Mapping[str, Any],
    training_result: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, CodecFreeIncrementOperator3D], dict[str, Any]]:
    parent, parent_config, parent_record = verify_and_load_parent(
        training_manifest, device=device
    )
    checkpoint = Path(str(training_result["best_checkpoint"]["path"]))
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    candidate, candidate_config = build_model(training_manifest["architecture"])
    if (
        payload.get("family") != "c5p"
        or payload.get("stage") != "four_step_detached_feedback_finetune"
        or int(payload.get("seed", -1)) != 1702
        or payload.get("parent_checkpoint") != training_manifest["parent"]["checkpoint"]
        or payload.get("loss") != training_manifest["loss"]
    ):
        raise ValueError("selected feedback checkpoint payload differs")
    config_validation = validate_parent_config(
        payload.get("config", {}), candidate_config.to_record()
    )
    candidate = candidate.to(device)
    candidate.load_state_dict(payload["model"], strict=True)
    bitwise = all(
        torch.equal(payload["model"][name].to(device), candidate.state_dict()[name])
        for name in payload["model"]
    )
    if not bitwise:
        raise AssertionError("selected feedback checkpoint did not load bitwise")
    if parent_config.to_record() != candidate_config.to_record():
        raise ValueError("parent and feedback architectures differ")
    parent.eval()
    candidate.eval()
    return {
        "pre_feedback_parent": parent,
        "four_step_feedback_finetuned": candidate,
    }, {
        "pre_feedback_parent": parent_record,
        "four_step_feedback_finetuned": {
            "checkpoint": dict(training_result["best_checkpoint"]),
            "checkpoint_reload_bitwise": True,
            "checkpoint_config_validation": config_validation,
            "parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
        },
    }


def create_forecast_file(
    path: Path,
    *,
    paper0_commit: str,
    manifest_sha256: str,
    training_result_sha256: str,
) -> h5py.File:
    if path.exists():
        raise FileExistsError(path)
    handle = h5py.File(path, "w", libver="latest", track_order=True)
    handle.attrs.update(
        {
            "schema_version": 1,
            "scope": SCOPE,
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "zperiod": 5,
            "stored_value": "standardized_terminal_state_delta_from_current",
            "paper0_commit": paper0_commit,
            "manifest_sha256": manifest_sha256,
            "training_result_sha256": training_result_sha256,
            "fields": json.dumps(list(FIELDS)),
            "methods": json.dumps(list(METHODS)),
        }
    )
    for horizon in HORIZONS:
        count = VALIDATION_STOP - horizon - VALIDATION_START
        current = np.arange(VALIDATION_START, VALIDATION_STOP - horizon, dtype=np.int64)
        group = handle.create_group(f"horizon_{horizon}")
        group.attrs["horizon_saved_frames"] = horizon
        group.attrs["pair_count"] = count
        group.create_dataset("current_frame", data=current, track_times=False)
        group.create_dataset("target_frame", data=current + horizon, track_times=False)
        for method in METHODS:
            _create_delta_dataset(group, method, count)
    return handle


def _append_state_rows(
    rows: list[dict[str, Any]],
    frame_rmse: np.ndarray,
    *,
    method: str,
    horizon: int,
    current_frames: np.ndarray,
) -> None:
    for offset, current in enumerate(current_frames):
        for field_index, field in enumerate(FIELDS):
            rows.append(
                {
                    "method": method,
                    "horizon_frames": horizon,
                    "current_frame": int(current),
                    "target_frame": int(current + horizon),
                    "field": field,
                    "standardized_rmse": float(frame_rmse[offset, field_index]),
                }
            )


def score_horizon(
    *,
    handle: h5py.File,
    models: Mapping[str, CodecFreeIncrementOperator3D],
    states: np.ndarray,
    horizon: int,
    normalization: Any,
    truth_transport: Mapping[str, Mapping[str, np.ndarray]],
    geometry: Any,
    device: torch.device,
    batch_size: int,
    state_rows: list[dict[str, Any]],
    examples: dict[str, np.ndarray],
    example_start: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_frames = np.arange(VALIDATION_START, VALIDATION_STOP - horizon, dtype=np.int64)
    target_frames = current_frames + horizon
    current = _frame_slice(states, current_frames)
    truth = _frame_slice(states, target_frames)
    _, persistence_mse, _ = _persistence_records(states, horizon=horizon)
    state_accumulators = {method: FieldErrorAccumulator.empty() for method in METHODS}
    matched = {
        method: MatchedCodecAccumulator(spec=SPEC, n_z=88, zperiod=5)
        for method in METHODS
    }
    transport = TransportComparisonAccumulator(
        comparisons={f"truth_vs_{method}": ("truth", method) for method in METHODS}
    )
    group = handle[f"horizon_{horizon}"]
    with torch.inference_mode():
        for start in range(0, len(current_frames), batch_size):
            stop = min(len(current_frames), start + batch_size)
            current_batch = np.ascontiguousarray(current[start:stop], dtype=np.float32)
            truth_batch = np.ascontiguousarray(truth[start:stop], dtype=np.float32)
            current_tensor = torch.from_numpy(current_batch).to(device)
            physical_truth = decode_batch(normalization, truth_batch)
            path_outputs: dict[str, Any] = {
                "truth": _transport_slice(truth_transport, start, stop)
            }
            example_indices = np.flatnonzero(
                current_frames[start:stop] == example_start
            )
            for method, model in models.items():
                terminal = autoregressive_forecast_path(
                    model, current_tensor, step=1, horizon=horizon
                )[-1]
                candidate = np.ascontiguousarray(
                    terminal.float().cpu().numpy(), dtype=np.float32
                )
                if not np.all(np.isfinite(candidate)):
                    raise ValueError("feedback evaluation forecast is non-finite")
                group[method][start:stop] = candidate - current_batch
                frame_rmse = state_accumulators[method].update(candidate, truth_batch)
                _append_state_rows(
                    state_rows,
                    frame_rmse,
                    method=method,
                    horizon=horizon,
                    current_frames=current_frames[start:stop],
                )
                physical_candidate = decode_batch(normalization, candidate)
                matched[method].update(
                    truth_batch,
                    candidate,
                    physical_truth,
                    physical_candidate,
                )
                path_outputs[method] = transport_from_model88(
                    physical_candidate, geometry
                )
                if example_indices.size:
                    examples[f"h{horizon}_{method}"] = np.asarray(
                        physical_candidate[int(example_indices[0])], dtype=np.float32
                    )
            if example_indices.size:
                local = int(example_indices[0])
                examples[f"h{horizon}_current"] = decode_batch(
                    normalization, current_batch
                )[local].astype(np.float32)
                examples[f"h{horizon}_truth"] = physical_truth[local].astype(np.float32)
            transport.update(path_outputs)
    state = {
        method: accumulator.finalize(persistence_mse=persistence_mse)
        for method, accumulator in state_accumulators.items()
    }
    physics = {
        "field_spectral_cross": {
            method: _json_safe(accumulator.finalize())
            for method, accumulator in matched.items()
        },
        "transport": _json_safe(transport.finalize()),
    }
    return state, physics


def _median_absolute_log_power_error(physics: Mapping[str, Any]) -> float:
    values = []
    for field in FIELDS:
        for band in BANDS:
            ratio = float(physics["field_band_summaries"][field][band]["power_ratio"])
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError("power ratio must be finite and positive")
            values.append(abs(math.log(ratio)))
    return float(np.median(values))


def _mean_separatrix_relative_l2(
    transport: Mapping[str, Any], *, method: str
) -> float:
    quantities = transport["comparisons"][f"truth_vs_{method}"]["quantities"]
    return float(
        np.mean(
            [
                float(quantities[quantity]["separatrix"]["metrics"]["relative_l2"])
                for quantity in QUANTITIES
            ]
        )
    )


def physics_preservation_decision(
    *,
    by_horizon: Mapping[str, Any],
    state_pilot_passed: bool,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    horizon_records: dict[str, Any] = {}
    all_power = True
    all_transport = True
    for horizon in HORIZONS:
        physics = by_horizon[str(horizon)]
        parent_power = _median_absolute_log_power_error(
            physics["field_spectral_cross"]["pre_feedback_parent"]
        )
        candidate_power = _median_absolute_log_power_error(
            physics["field_spectral_cross"]["four_step_feedback_finetuned"]
        )
        parent_transport = _mean_separatrix_relative_l2(
            physics["transport"], method="pre_feedback_parent"
        )
        candidate_transport = _mean_separatrix_relative_l2(
            physics["transport"], method="four_step_feedback_finetuned"
        )
        power_ratio = (
            candidate_power / parent_power
            if parent_power > 0.0
            else (1.0 if candidate_power == 0.0 else math.inf)
        )
        transport_ratio = (
            candidate_transport / parent_transport
            if parent_transport > 0.0
            else (1.0 if candidate_transport == 0.0 else math.inf)
        )
        power_passed = power_ratio <= 1.0 + float(
            gates["maximum_absolute_log_power_ratio_error_increase_fraction"]
        )
        transport_passed = transport_ratio <= 1.0 + float(
            gates["maximum_mean_separatrix_relative_l2_increase_fraction"]
        )
        all_power = all_power and power_passed
        all_transport = all_transport and transport_passed
        horizon_records[str(horizon)] = {
            "parent_median_absolute_log_power_ratio_error": parent_power,
            "candidate_median_absolute_log_power_ratio_error": candidate_power,
            "candidate_over_parent_power_error": power_ratio,
            "power_preservation_passed": power_passed,
            "parent_mean_separatrix_relative_l2": parent_transport,
            "candidate_mean_separatrix_relative_l2": candidate_transport,
            "candidate_over_parent_separatrix_error": transport_ratio,
            "separatrix_transport_preservation_passed": transport_passed,
        }
    decision_gates = {
        "state_pilot_passed": bool(state_pilot_passed),
        "power_preserved_at_both_horizons": all_power,
        "separatrix_transport_preserved_at_both_horizons": all_transport,
    }
    return {
        "by_horizon": horizon_records,
        "gates": decision_gates,
        "advance_to_confirmation_seeds": all(decision_gates.values()),
    }


def _write_state_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
        if "85606" in str(path).lower():
            raise ValueError("held-out 85606 paths are prohibited")
    if args.output.exists():
        raise FileExistsError(args.output)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 checkout commit differs")
    manifest = load_strict_json(args.manifest)
    authorize_evaluation_manifest(
        manifest,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
    )
    training_manifest, training_result, training_result_lock = authorize_training_result(
        manifest
    )
    args.output.mkdir(parents=True)

    evidence_manifest_path = verify_locked_file(
        manifest.get("bounded_evidence_manifest", {}), label="bounded evidence manifest"
    )
    evidence_manifest = load_strict_json(evidence_manifest_path)
    evidence = evidence_manifest["evidence"]
    native_path = verify_locked_file(
        evidence["native_truth_result"], label="native truth result"
    )
    geometry_manifest_path = verify_locked_file(
        evidence["geometry_manifest"], label="geometry manifest"
    )
    geometry_path = verify_locked_file(evidence["geometry"], label="geometry")
    native_catalog = NativeTruthCatalog(load_strict_json(native_path))
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    catalog = load_official_catalog(args.artifact_root)
    states = load_validation_states(catalog)

    if not torch.cuda.is_available():
        raise RuntimeError("feedback evaluation requires an allocated CUDA GPU")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    models, model_records = load_models(
        training_manifest, training_result, device=device
    )

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("online W&B is required") from error
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="old-85604-four-step-feedback-evaluation",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "codec-free",
            "four-step-feedback",
            "bounded-rollout",
            "spectra",
            "cross-phase",
            "transport",
            "evaluation-only",
        ),
    )
    api = wandb.Api(timeout=30)
    if not api.api_key:
        raise RuntimeError("W&B API key is absent")
    viewer = api.viewer
    if str(getattr(viewer, "entity", "")) != spec.entity:
        raise RuntimeError("authenticated W&B entity differs")
    tracking_directory = args.output / "wandb"
    tracking_directory.mkdir()
    run = wandb.init(
        entity=spec.entity,
        project=spec.project,
        group=spec.group,
        name=spec.run_name,
        id=spec.run_id,
        resume="never",
        job_type=spec.job_type,
        tags=list(spec.tags),
        config={
            "scope": SCOPE,
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "horizons": list(HORIZONS),
            "methods": list(METHODS),
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "paper0_commit": args.paper0_commit,
            "manifest_sha256": args.manifest_sha256,
            "training_result": training_result_lock,
            "physics_derived_training_loss": False,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")

    started = time.perf_counter()
    state_by_horizon: dict[str, Any] = {}
    physics_by_horizon: dict[str, Any] = {}
    common: dict[str, Any] = {}
    state_rows: list[dict[str, Any]] = []
    examples: dict[str, np.ndarray] = {}
    forecast_path = args.output / "terminal_forecast_deltas.h5"
    try:
        with create_forecast_file(
            forecast_path,
            paper0_commit=args.paper0_commit,
            manifest_sha256=args.manifest_sha256,
            training_result_sha256=training_result_lock["sha256"],
        ) as handle:
            for horizon in HORIZONS:
                truth_transport = _native_truth_transport(
                    native_catalog,
                    start=VALIDATION_START + horizon,
                    stop=VALIDATION_STOP,
                    geometry=geometry,
                )
                common[str(horizon)] = score_common_persistence(
                    states=states,
                    horizon=horizon,
                    normalization=catalog.normalization,
                    truth_transport=truth_transport,
                    geometry=geometry,
                    batch_size=4,
                    examples=examples,
                    example_start=560,
                )
                state, physics = score_horizon(
                    handle=handle,
                    models=models,
                    states=states,
                    horizon=horizon,
                    normalization=catalog.normalization,
                    truth_transport=truth_transport,
                    geometry=geometry,
                    device=device,
                    batch_size=4,
                    state_rows=state_rows,
                    examples=examples,
                    example_start=560,
                )
                state_by_horizon[str(horizon)] = state
                physics_by_horizon[str(horizon)] = physics
                run.log(
                    {
                        f"state/h{horizon}/parent_skill": state[
                            "pre_feedback_parent"
                        ]["mean_field_persistence_relative_skill"],
                        f"state/h{horizon}/feedback_skill": state[
                            "four_step_feedback_finetuned"
                        ]["mean_field_persistence_relative_skill"],
                        f"transport/h{horizon}/feedback_particle_separatrix_relative_l2": physics[
                            "transport"
                        ]["comparisons"]["truth_vs_four_step_feedback_finetuned"][
                            "quantities"
                        ]["particle"]["separatrix"]["metrics"]["relative_l2"],
                    },
                    step=horizon,
                )
            handle.flush()

        decision = physics_preservation_decision(
            by_horizon=physics_by_horizon,
            state_pilot_passed=bool(training_result["state_pilot_passed"]),
            gates=manifest["physics_preservation_gates"],
        )
        state_path = args.output / "state_metrics.json"
        physics_path = args.output / "physics_metrics.json"
        state_csv = args.output / "per_target_state_rmse.csv"
        examples_path = args.output / "example_physical_fields_start560.npz"
        atomic_json(
            state_path,
            {
                "schema_version": 1,
                "scope": SCOPE,
                "development_run": "85604",
                "held_out_85606_read": False,
                "new_nersc_data_read": False,
                "guard_frames_read": False,
                "physics_derived_metric": False,
                "by_horizon": state_by_horizon,
            },
        )
        atomic_json(
            physics_path,
            {
                "schema_version": 1,
                "scope": SCOPE,
                "development_run": "85604",
                "held_out_85606_read": False,
                "new_nersc_data_read": False,
                "guard_frames_read": False,
                "training_performed": False,
                "physics_derived_loss_used": False,
                "zperiod": 5,
                "mode_mapping": "n=5k",
                "fields": list(FIELDS),
                "common_persistence": common,
                "by_horizon": physics_by_horizon,
                "decision": decision,
            },
        )
        _write_state_csv(state_csv, state_rows)
        if examples_path.exists():
            raise FileExistsError(examples_path)
        np.savez_compressed(examples_path, **examples)
        result = {
            "schema_version": 1,
            "scope": SCOPE,
            "status": "feedback_pilot_state_and_physics_scored",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "physics_derived_loss_used": False,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "training_result": training_result_lock,
            "models": model_records,
            "forecast": {
                "path": str(forecast_path),
                "sha256": sha256_path(forecast_path),
                "stored_value": "standardized_terminal_state_delta_from_current",
            },
            "state_metrics": {"path": str(state_path), "sha256": sha256_path(state_path)},
            "physics_metrics": {
                "path": str(physics_path),
                "sha256": sha256_path(physics_path),
            },
            "per_target_state_rmse": {
                "path": str(state_csv),
                "sha256": sha256_path(state_csv),
                "row_count": len(state_rows),
            },
            "example_fields": {
                "path": str(examples_path),
                "sha256": sha256_path(examples_path),
                "start_frame": 560,
            },
            "decision": decision,
            "confirmation_seed_training_authorized": decision[
                "advance_to_confirmation_seeds"
            ],
            "wall_seconds_before_wandb_verification": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(device),
        }
        run.summary.update(
            {
                "final/status": result["status"],
                "final/state_pilot_passed": bool(training_result["state_pilot_passed"]),
                "final/advance_to_confirmation_seeds": decision[
                    "advance_to_confirmation_seeds"
                ],
                "scope/held_out_85606_read": False,
                "scope/new_nersc_data_read": False,
                "scope/training_performed": False,
                "scope/physics_derived_loss_used": False,
                "compute/wall_seconds": result[
                    "wall_seconds_before_wandb_verification"
                ],
            }
        )
        run_url = str(run.url)
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise

    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote_state = verify_finished_wandb_run(
        module=wandb,
        remote_path=remote_path,
        expected_id=spec.run_id,
    )
    tracking = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "spec": spec.to_record(),
        "authenticated_username": str(getattr(viewer, "username", "")),
        "wandb_version": wandb.__version__,
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_state_after_finish": remote_state,
        "forecast_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    atomic_json(args.output / "wandb.json", tracking)
    index = args.output / "artifact_sha256.txt"
    index.write_text(
        "".join(
            f"{sha256_path(path)}  {path.resolve(strict=True)}\n"
            for path in sorted(args.output.iterdir())
            if path.is_file() and path != index
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
