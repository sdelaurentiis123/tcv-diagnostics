#!/usr/bin/env python3
"""Score saved bounded rollouts with validated evaluation-only physics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import h5py
import numpy as np

from paper0.tools.evaluate_codec_free_bounded_rollout import (
    HORIZONS,
    SEEDS,
    VALIDATION_START,
    authorize_manifest,
    load_validation_states,
    verify_locked_file,
)
from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    verify_finished_wandb_run,
)
from tcv_diagnostics.bounded_rollout import FIELDS, method_schedule
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.codec_transport import (
    TransportComparisonAccumulator,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from tcv_diagnostics.matched_codec_metrics import (
    CodecViewSpec,
    MatchedCodecAccumulator,
)
from tcv_diagnostics.matched_o1_transport import (
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.resampling import periodic_resample_float32
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SCOPE = "post_ecrd_old_85604_bounded_rollout_physics"
SPEC = CodecViewSpec(
    name="old_85604_bounded_rollout_c5p",
    fields=FIELDS,
    spectral_fields=FIELDS,
    cross_pairs=(("Ne", "phi"), ("Pe", "phi"), ("Pi", "phi")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--state-result", type=Path, required=True)
    parser.add_argument("--state-result-sha256", required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--forecast-sha256", required=True)
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


def authorize_state_artifact(
    *,
    result_path: Path,
    result_sha256: str,
    forecast_path: Path,
    forecast_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    if sha256_path(result_path) != result_sha256:
        raise ValueError("bounded state result SHA-256 differs")
    result = load_strict_json(result_path)
    if (
        result.get("scope") != "post_ecrd_old_85604_bounded_rollout"
        or result.get("status") != "bounded_state_forecast_generated_and_scored"
        or result.get("development_run") != "85604"
        or result.get("held_out_85606_read") is not False
        or result.get("new_nersc_data_read") is not False
        or result.get("guard_frames_read") is not False
        or result.get("training_performed") is not False
        or result.get("checkpoint_selection_performed") is not False
        or result.get("physics_derived_loss_used") is not False
        or result.get("physics_diagnostics_scored") is not False
        or result.get("physics_scoring_authorized_next") is not True
        or Path(str(result.get("manifest", ""))) != manifest_path
        or result.get("manifest_sha256") != manifest_sha256
        or len(str(result.get("paper0_commit", ""))) != 40
    ):
        raise ValueError("bounded state result contract differs")
    forecast = result.get("forecast_artifact", {})
    if (
        Path(str(forecast.get("path", ""))) != forecast_path
        or forecast.get("sha256") != forecast_sha256
        or sha256_path(forecast_path) != forecast_sha256
        or forecast.get("stored_value")
        != "standardized_terminal_state_delta_from_current"
        or forecast.get("dtype") != "float32"
    ):
        raise ValueError("bounded forecast artifact identity differs")
    return result


def validate_forecast_schema(
    handle: h5py.File,
    *,
    paper0_commit: str,
    manifest_sha256: str,
) -> None:
    expected_attributes = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_bounded_rollout",
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "guard_frames_read": False,
        "zperiod": 5,
        "stored_value": "standardized_terminal_state_delta_from_current",
        "paper0_commit": paper0_commit,
        "manifest_sha256": manifest_sha256,
    }
    for name, expected in expected_attributes.items():
        actual = handle.attrs.get(name)
        if isinstance(expected, bool):
            actual = bool(actual)
        elif isinstance(expected, int):
            actual = int(actual)
        else:
            actual = str(actual)
        if actual != expected:
            raise ValueError(f"bounded forecast attribute {name!r} differs")
    if json.loads(str(handle.attrs["fields"])) != list(FIELDS):
        raise ValueError("bounded forecast fields differ")
    if set(handle) != {"horizon_4", "horizon_8"}:
        raise ValueError("bounded forecast horizon groups differ")
    for horizon in HORIZONS:
        group = handle[f"horizon_{horizon}"]
        count = 624 - horizon - 496
        current = np.arange(496, 624 - horizon, dtype=np.int64)
        if (
            int(group.attrs["horizon_saved_frames"]) != horizon
            or int(group.attrs["pair_count"]) != count
            or not np.array_equal(group["current_frame"][:], current)
            or not np.array_equal(group["target_frame"][:], current + horizon)
        ):
            raise ValueError(f"bounded horizon-{horizon} coordinates differ")
        expected_children = {
            "current_frame",
            "target_frame",
            *(f"seed_{seed}" for seed in SEEDS),
        }
        if set(group) != expected_children:
            raise ValueError(f"bounded horizon-{horizon} inventory differs")
        for seed in SEEDS:
            seed_group = group[f"seed_{seed}"]
            if set(seed_group) != set(method_schedule(horizon)):
                raise ValueError("bounded forecast method inventory differs")
            for method in method_schedule(horizon):
                dataset = seed_group[method]
                if (
                    dataset.shape != (count, len(FIELDS), 64, 32, 88)
                    or dataset.dtype != np.dtype("f4")
                    or dataset.compression != "gzip"
                    or dataset.compression_opts != 4
                    or not dataset.shuffle
                    or not dataset.fletcher32
                ):
                    raise ValueError("bounded forecast dataset schema differs")


def decode_batch(normalization: Any, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 5 or array.shape[1:] != (len(FIELDS), 64, 32, 88):
        raise ValueError("standardized C5P batch shape differs")
    physical = np.stack(
        [
            normalization.records[field].decode(array[:, index])
            for index, field in enumerate(FIELDS)
        ],
        axis=1,
    )
    if not np.all(np.isfinite(physical)):
        raise ValueError("decoded C5P batch is non-finite")
    return physical


def transport_from_model88(
    physical_model88: np.ndarray,
    geometry: Any,
) -> dict[str, dict[str, np.ndarray]]:
    values = np.asarray(physical_model88)
    if values.ndim != 5 or values.shape[1:] != (len(FIELDS), 64, 32, 88):
        raise ValueError("model-grid transport batch shape differs")
    native = periodic_resample_float32(
        np.asarray(values[:, :4], dtype=np.float32),
        81,
        axis=-1,
    ).astype(np.float64)
    return evaluate_transport_state(
        direct_pressure_transport_state(
            native[:, 0], native[:, 1], native[:, 2], native[:, 3]
        ),
        geometry,
    )


def _transport_slice(
    values: Mapping[str, Mapping[str, np.ndarray]],
    start: int,
    stop: int,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        quantity: {
            reduction: np.asarray(array[start:stop])
            for reduction, array in reductions.items()
        }
        for quantity, reductions in values.items()
    }


def _native_truth_transport(
    native_catalog: NativeTruthCatalog,
    *,
    start: int,
    stop: int,
    geometry: Any,
) -> dict[str, dict[str, np.ndarray]]:
    state = native_catalog.read(start, stop, fields=("Ne", "Pe", "Pi", "phi"))
    return evaluate_transport_state(
        direct_pressure_transport_state(
            state["Ne"], state["Pe"], state["Pi"], state["phi"]
        ),
        geometry,
    )


def score_common_persistence(
    *,
    states: np.ndarray,
    horizon: int,
    normalization: Any,
    truth_transport: Mapping[str, Mapping[str, np.ndarray]],
    geometry: Any,
    batch_size: int,
    examples: dict[str, np.ndarray],
    example_start: int,
) -> dict[str, Any]:
    current_frames = np.arange(496, 624 - horizon, dtype=np.int64)
    current = states[current_frames - VALIDATION_START]
    truth = states[current_frames + horizon - VALIDATION_START]
    matched = MatchedCodecAccumulator(spec=SPEC, n_z=88, zperiod=5)
    transport = TransportComparisonAccumulator(
        comparisons={"truth_vs_persistence": ("truth", "persistence")}
    )
    for start in range(0, len(current_frames), batch_size):
        stop = min(len(current_frames), start + batch_size)
        physical_truth = decode_batch(normalization, truth[start:stop])
        physical_current = decode_batch(normalization, current[start:stop])
        matched.update(
            truth[start:stop],
            current[start:stop],
            physical_truth,
            physical_current,
        )
        transport.update(
            {
                "truth": _transport_slice(truth_transport, start, stop),
                "persistence": transport_from_model88(physical_current, geometry),
            }
        )
        indices = np.flatnonzero(current_frames[start:stop] == example_start)
        if indices.size:
            local = int(indices[0])
            examples[f"h{horizon}_current"] = np.asarray(
                physical_current[local], dtype=np.float32
            )
            examples[f"h{horizon}_truth"] = np.asarray(
                physical_truth[local], dtype=np.float32
            )
    return {
        "field_spectral_cross": _json_safe(matched.finalize()),
        "transport": _json_safe(transport.finalize()),
    }


def score_seed_horizon(
    *,
    handle: h5py.File,
    states: np.ndarray,
    seed: int,
    horizon: int,
    normalization: Any,
    truth_transport: Mapping[str, Mapping[str, np.ndarray]],
    geometry: Any,
    batch_size: int,
    examples: dict[str, np.ndarray],
    example_start: int,
) -> dict[str, Any]:
    current_frames = np.arange(496, 624 - horizon, dtype=np.int64)
    current = states[current_frames - VALIDATION_START]
    truth = states[current_frames + horizon - VALIDATION_START]
    methods = method_schedule(horizon)
    matched = {
        method: MatchedCodecAccumulator(spec=SPEC, n_z=88, zperiod=5)
        for method in methods
    }
    comparisons = {f"truth_vs_{method}": ("truth", method) for method in methods}
    transport = TransportComparisonAccumulator(comparisons=comparisons)
    group = handle[f"horizon_{horizon}/seed_{seed}"]
    for start in range(0, len(current_frames), batch_size):
        stop = min(len(current_frames), start + batch_size)
        physical_truth = decode_batch(normalization, truth[start:stop])
        path_outputs: dict[str, Any] = {
            "truth": _transport_slice(truth_transport, start, stop)
        }
        indices = np.flatnonzero(current_frames[start:stop] == example_start)
        for method in methods:
            delta = np.asarray(group[method][start:stop], dtype=np.float32)
            if not np.all(np.isfinite(delta)):
                raise ValueError("bounded forecast delta is non-finite")
            candidate = current[start:stop] + delta
            physical_candidate = decode_batch(normalization, candidate)
            matched[method].update(
                truth[start:stop],
                candidate,
                physical_truth,
                physical_candidate,
            )
            path_outputs[method] = transport_from_model88(physical_candidate, geometry)
            if indices.size:
                examples[f"h{horizon}_seed{seed}_{method}"] = np.asarray(
                    physical_candidate[int(indices[0])], dtype=np.float32
                )
        transport.update(path_outputs)
    return {
        "field_spectral_cross": {
            method: _json_safe(accumulator.finalize())
            for method, accumulator in matched.items()
        },
        "transport": _json_safe(transport.finalize()),
    }


def main() -> None:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.manifest,
        args.state_result,
        args.forecast,
        args.output,
        args.paper0_root,
    ):
        assert_development_path(path)
        if "85606" in str(path).lower():
            raise ValueError("held-out 85606 paths are prohibited")
    if args.output.exists():
        raise FileExistsError(args.output)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 checkout commit differs")
    state_result = authorize_state_artifact(
        result_path=args.state_result,
        result_sha256=args.state_result_sha256,
        forecast_path=args.forecast,
        forecast_sha256=args.forecast_sha256,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
    )
    manifest = load_strict_json(args.manifest)
    forecast_paper0_commit = str(state_result["paper0_commit"])
    authorize_manifest(
        manifest,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        paper0_commit=forecast_paper0_commit,
    )
    args.output.mkdir(parents=True)

    evidence = manifest["evidence"]
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
        job_type="old-85604-bounded-rollout-physics",
        tags=(
            "paper0",
            "85604",
            "old-data",
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
            "seeds": list(SEEDS),
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "paper0_commit": args.paper0_commit,
            "forecast_paper0_commit": forecast_paper0_commit,
            "manifest_sha256": args.manifest_sha256,
            "forecast_sha256": args.forecast_sha256,
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
    common: dict[str, Any] = {}
    by_seed: dict[str, Any] = {str(seed): {} for seed in SEEDS}
    examples: dict[str, np.ndarray] = {}
    try:
        with h5py.File(args.forecast, "r") as handle:
            validate_forecast_schema(
                handle,
                paper0_commit=state_result["paper0_commit"],
                manifest_sha256=args.manifest_sha256,
            )
            for horizon in HORIZONS:
                target_start = 496 + horizon
                truth_transport = _native_truth_transport(
                    native_catalog,
                    start=target_start,
                    stop=624,
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
                for seed in SEEDS:
                    record = score_seed_horizon(
                        handle=handle,
                        states=states,
                        seed=seed,
                        horizon=horizon,
                        normalization=catalog.normalization,
                        truth_transport=truth_transport,
                        geometry=geometry,
                        batch_size=4,
                        examples=examples,
                        example_start=560,
                    )
                    by_seed[str(seed)][str(horizon)] = record
                    metrics: dict[str, float] = {}
                    for method in method_schedule(horizon):
                        physics = record["field_spectral_cross"][method]
                        metrics[
                            f"physics/h{horizon}/{method}/Pe_phi_k4_5_phase_error_deg"
                        ] = physics["cross_field_band_summaries"]["Pe-phi"]["k4_5"][
                            "truth_cross_amplitude_weighted_absolute_phase_error_degrees"
                        ]
                        transport = record["transport"]["comparisons"][
                            f"truth_vs_{method}"
                        ]["quantities"]
                        metrics[
                            f"transport/h{horizon}/{method}/particle_separatrix_relative_l2"
                        ] = transport["particle"]["separatrix"]["metrics"][
                            "relative_l2"
                        ]
                    run.log(metrics, step=(seed - 1700) * 10 + horizon)

        physics_metrics = {
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
            "spectral_bands": {
                "k1_3": {"stored_k": [1, 3], "full_torus_n": [5, 15]},
                "k4_5": {"stored_k": [4, 5], "full_torus_n": [20, 25]},
                "k6_7": {"stored_k": [6, 7], "full_torus_n": [30, 35]},
            },
            "common_persistence": common,
            "by_seed": by_seed,
        }
        metrics_path = args.output / "physics_metrics.json"
        atomic_json(metrics_path, physics_metrics)
        examples_path = args.output / "example_physical_fields_start560.npz"
        if examples_path.exists():
            raise FileExistsError(examples_path)
        np.savez_compressed(examples_path, **examples)

        result = {
            "schema_version": 1,
            "scope": SCOPE,
            "status": "bounded_rollout_physics_scored",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "physics_derived_loss_used": False,
            "paper0_commit": args.paper0_commit,
            "forecast_paper0_commit": forecast_paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "state_result": {
                "path": str(args.state_result),
                "sha256": args.state_result_sha256,
            },
            "forecast": {
                "path": str(args.forecast),
                "sha256": args.forecast_sha256,
                "mutated_or_regenerated": False,
            },
            "physics_metrics": {
                "path": str(metrics_path),
                "sha256": sha256_path(metrics_path),
            },
            "example_fields": {
                "path": str(examples_path),
                "sha256": sha256_path(examples_path),
                "start_frame": 560,
            },
            "wall_seconds_before_wandb_verification": time.perf_counter() - started,
        }
        run.summary.update(
            {
                "final/status": result["status"],
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
