#!/usr/bin/env python3
"""Score the frozen C5P/E6B bounded pair in one common physical view."""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import h5py
import numpy as np

from paper0.tools.freeze_matched_state_physics_scoring import (
    SCOPE as MANIFEST_SCOPE,
)
from paper0.tools.generate_matched_state_bounded_forecasts import (
    FAMILIES,
    HORIZONS,
    SCOPE as GENERATION_SCOPE,
)
from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    verify_finished_wandb_run,
)
from tcv_diagnostics.bounded_rollout import method_schedule
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.codec_transport import (
    TRANSPORT_QUANTITIES,
    TransportComparisonAccumulator,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from tcv_diagnostics.matched_codec_evaluation import (
    COMMON_CROSS_PAIRS,
    COMMON_FIELDS,
    decode_physical_batch,
    encode_physical_batch,
)
from tcv_diagnostics.matched_codec_metrics import (
    MODE_BANDS,
    CodecViewSpec,
    MatchedCodecAccumulator,
)
from tcv_diagnostics.matched_o1_transport import (
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import (
    FAMILY_FIELDS,
    VOLUME_SHAPE,
    CodecFrameDataset,
    load_official_catalog,
)
from tcv_diagnostics.resampling import periodic_resample_float32
from tcv_diagnostics.state_completeness import soft_floor
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SCOPE = "post_ecrd_old_85604_matched_state_bounded_physics"
VALIDATION_START = 496
VALIDATION_STOP = 624
NATIVE_SHAPE = (64, 32, 81)
BATCH_SIZE = 4
SPEC = CodecViewSpec(
    name="old_85604_matched_state_common_C5P",
    fields=COMMON_FIELDS,
    spectral_fields=COMMON_FIELDS,
    cross_pairs=COMMON_CROSS_PAIRS,
)
BAND_BOUNDS = {label: (low, high) for label, low, high in MODE_BANDS}
EXPECTED_PRIMARY_COUNTS = {
    "separatrix_transport_relative_l2": 84,
    "complex_cross_spectrum_relative_l2": 189,
    "shared_state_standardized_rmse": 63,
    "spectral_power_absolute_log_ratio": 315,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def locked_path(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return path


def authorize_manifest(
    path: Path,
    digest: str,
) -> dict[str, Any]:
    if sha256_path(path) != digest:
        raise ValueError("paired physics manifest SHA-256 differs")
    manifest = load_strict_json(path)
    expected = {
        "scope": MANIFEST_SCOPE,
        "status": "frozen_after_causal_exact_phi_before_paired_physics",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_read": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "physics_derived_training_loss_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "steering_allowed": False,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "state_views": list(FAMILIES),
        "common_fields": list(COMMON_FIELDS),
        "wandb_required": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("paired physics manifest contract differs")
    evaluation = manifest.get("evaluation", {})
    expected_blocks = {
        "4": [[500, 541], [541, 582], [582, 624]],
        "8": [[504, 544], [544, 584], [584, 624]],
    }
    if (
        evaluation.get("horizons") != list(HORIZONS)
        or evaluation.get("target_frame_blocks") != expected_blocks
        or evaluation.get("physics_diagnostics_are_evaluation_only") is not True
    ):
        raise ValueError("paired physics block contract differs")
    for label, record in manifest.get("dependencies", {}).items():
        locked_path(record, label=label)
    for family in FAMILIES:
        result_record = manifest.get("generation_results", {}).get(family, {})
        result_path = locked_path(result_record, label=f"{family} generation result")
        result = load_strict_json(result_path)
        if (
            result.get("scope") != GENERATION_SCOPE
            or result.get("status") != "completed"
            or result.get("family") != family
            or result.get("held_out_85606_read") is not False
            or result.get("new_nersc_data_read") is not False
            or result.get("target_truth_used_during_generation") is not False
        ):
            raise ValueError(f"{family} generation result no longer qualifies")
        forecast = result_record.get("forecast", {})
        if forecast != result.get("forecast"):
            raise ValueError(f"{family} forecast lock differs from its result")
        locked_path(forecast, label=f"{family} bounded forecast")
    exact_path = locked_path(
        manifest.get("exact_phi_result", {}), label="exact-phi result"
    )
    exact = load_strict_json(exact_path)
    if (
        exact.get("status") != "completed"
        or exact.get("target_truth_phi_read") is not False
        or exact.get("truth_layout") is not False
        or exact.get("candidate_count") != 7
        or exact.get("paired_common_view_physics_scoring_authorized") is not True
    ):
        raise ValueError("exact-phi result no longer authorizes scoring")
    return manifest


def validate_forecast_schema(handle: h5py.File, *, family: str) -> None:
    expected_attributes = {
        "schema_version": 1,
        "scope": GENERATION_SCOPE,
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "family": family,
        "zperiod": 5,
        "target_truth_used_during_generation": False,
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
            raise ValueError(f"{family} forecast attribute {name!r} differs")
    if set(handle) != {"horizon_4", "horizon_8"}:
        raise ValueError(f"{family} forecast horizon inventory differs")
    channels = len(FAMILY_FIELDS[family])
    for horizon in HORIZONS:
        count = VALIDATION_STOP - VALIDATION_START - horizon
        group = handle[f"horizon_{horizon}"]
        current = np.arange(VALIDATION_START, VALIDATION_STOP - horizon)
        target = current + horizon
        if (
            set(group) != {"coordinates", "methods"}
            or not np.array_equal(group["coordinates/current_frame"][:], current)
            or not np.array_equal(group["coordinates/target_frame"][:], target)
            or set(group["methods"]) != set(method_schedule(horizon))
        ):
            raise ValueError(f"{family} horizon-{horizon} coordinates differ")
        for method in method_schedule(horizon):
            method_group = group[f"methods/{method}"]
            expected_children = {"volume", "boundary_Bphi"} if family == "e6b" else {"volume"}
            if set(method_group) != expected_children:
                raise ValueError(f"{family} {method} output inventory differs")
            volume = method_group["volume"]
            if volume.shape != (count, channels, *VOLUME_SHAPE) or volume.dtype != np.dtype("f4"):
                raise ValueError(f"{family} {method} volume schema differs")
            if family == "e6b" and method_group["boundary_Bphi"].shape != (count, 2, 32):
                raise ValueError("E6B predicted-boundary schema differs")


def exact_phi_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Path]]:
    exact_path = Path(str(manifest["exact_phi_result"]["path"]))
    exact = load_strict_json(exact_path)
    result: dict[str, dict[str, Path]] = {}
    for record in exact["outputs"]:
        candidate = locked_path(record["candidate"], label="E6B candidate")
        phi = locked_path(record["derived_phi"], label="derived exact phi")
        elliptic = locked_path(record["elliptic_result"], label="elliptic result")
        elliptic_record = load_strict_json(elliptic)
        if (
            elliptic_record.get("truth_layout") is not False
            or elliptic_record.get("truth_replay_gate") is not None
            or Path(str(elliptic_record["source_input"]["path"]))
            != candidate.resolve()
        ):
            raise ValueError("exact-phi output provenance differs")
        result[candidate.name] = {"candidate": candidate, "phi": phi}
    if len(result) != 7:
        raise ValueError("exact-phi candidate index does not contain seven outputs")
    return result


def load_validation_states(catalog: Any) -> np.ndarray:
    dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="validation",
        frames=range(VALIDATION_START, VALIDATION_STOP),
        augment=False,
        seed=0,
        return_physical=False,
    )
    try:
        states = np.empty(
            (VALIDATION_STOP - VALIDATION_START, len(COMMON_FIELDS), *VOLUME_SHAPE),
            dtype=np.float32,
        )
        for index in range(len(dataset)):
            item = dataset[index]
            if int(item["frame_index"]) != VALIDATION_START + index:
                raise ValueError("validation frame order differs")
            states[index] = item["volume"]
    finally:
        dataset.close()
    if not np.all(np.isfinite(states)):
        raise ValueError("validation C5P state is non-finite")
    return states


def transport_from_model88(
    common_physical: np.ndarray,
    geometry: Any,
) -> dict[str, dict[str, np.ndarray]]:
    values = np.asarray(common_physical)
    if values.ndim != 5 or values.shape[1:] != (len(COMMON_FIELDS), *VOLUME_SHAPE):
        raise ValueError("common physical batch shape differs")
    native = periodic_resample_float32(
        np.asarray(values[:, :4], dtype=np.float32), 81, axis=-1
    ).astype(np.float64)
    return evaluate_transport_state(
        direct_pressure_transport_state(
            native[:, 0], native[:, 1], native[:, 2], native[:, 3]
        ),
        geometry,
    )


def transport_from_native(
    *,
    ne: np.ndarray,
    pe: np.ndarray,
    pi: np.ndarray,
    phi: np.ndarray,
    geometry: Any,
) -> dict[str, dict[str, np.ndarray]]:
    return evaluate_transport_state(
        direct_pressure_transport_state(ne, pe, pi, phi), geometry
    )


def _transport_slice(
    values: Mapping[str, Mapping[str, np.ndarray]], start: int, stop: int
) -> dict[str, dict[str, np.ndarray]]:
    return {
        quantity: {
            reduction: np.asarray(array[start:stop])
            for reduction, array in reductions.items()
        }
        for quantity, reductions in values.items()
    }


def e6b_common_physical(
    e6b_physical: np.ndarray,
    phi_model88: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    state = np.asarray(e6b_physical, dtype=np.float64)
    phi = np.asarray(phi_model88, dtype=np.float64)
    if state.ndim != 5 or state.shape[1:] != (6, *VOLUME_SHAPE):
        raise ValueError("E6B physical state shape differs")
    if phi.shape != (state.shape[0], *VOLUME_SHAPE):
        raise ValueError("E6B exact-phi model-grid shape differs")
    density = state[:, 0]
    floor = soft_floor(density, 1.0e-7)
    velocity = state[:, 4] / (2.0 * floor)
    common = np.stack(
        [density, state[:, 1], state[:, 2], phi, velocity], axis=1
    )
    if not np.all(np.isfinite(common)):
        raise ValueError("E6B common view is non-finite")
    return common, {
        "cell_count": int(density.size),
        "nonpositive_density_count": int(np.count_nonzero(density <= 0.0)),
        "below_density_floor_count": int(np.count_nonzero(density < 1.0e-7)),
        "minimum_density": float(np.min(density)),
        "maximum_absolute_derived_Vi": float(np.max(np.abs(velocity))),
    }


class PairAccumulator:
    def __init__(self) -> None:
        self.matched = MatchedCodecAccumulator(spec=SPEC, n_z=88, zperiod=5)
        self.transport = TransportComparisonAccumulator(
            comparisons={"truth_vs_candidate": ("truth", "candidate")}
        )

    def update(
        self,
        *,
        truth_standardized: np.ndarray,
        candidate_standardized: np.ndarray,
        truth_physical: np.ndarray,
        candidate_physical: np.ndarray,
        truth_transport: Mapping[str, Mapping[str, np.ndarray]],
        candidate_transport: Mapping[str, Mapping[str, np.ndarray]],
    ) -> None:
        self.matched.update(
            truth_standardized,
            candidate_standardized,
            truth_physical,
            candidate_physical,
        )
        self.transport.update(
            {"truth": truth_transport, "candidate": candidate_transport}
        )

    def finalize(self) -> dict[str, Any]:
        return {
            "field_spectral_cross": self.matched.finalize(),
            "transport": self.transport.finalize(),
        }


def complex_cross_band_relative_l2(
    metrics: Mapping[str, Any], pair: str, band: str
) -> float:
    if band not in BAND_BOUNDS:
        raise KeyError(band)
    curve = metrics["cross_field_curves_physical"][pair]
    truth = np.asarray(curve["truth_cross_spectrum_sum"]["real"]) + 1j * np.asarray(
        curve["truth_cross_spectrum_sum"]["imag"]
    )
    candidate = np.asarray(
        curve["reconstruction_cross_spectrum_sum"]["real"]
    ) + 1j * np.asarray(curve["reconstruction_cross_spectrum_sum"]["imag"])
    low, high = BAND_BOUNDS[band]
    selected = slice(low, high + 1)
    denominator = float(np.sum(np.abs(truth[selected]) ** 2))
    if denominator <= 0.0:
        raise ValueError(f"zero truth cross-spectrum norm for {pair} {band}")
    numerator = float(np.sum(np.abs(candidate[selected] - truth[selected]) ** 2))
    return math.sqrt(numerator / denominator)


def primary_rows(
    *,
    family: str,
    horizon: int,
    method: str,
    block_index: int,
    target_interval: tuple[int, int],
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    physics = metrics["field_spectral_cross"]
    transport = metrics["transport"]["comparisons"]["truth_vs_candidate"]
    base = {
        "family": family,
        "horizon": horizon,
        "method": method,
        "block_index": block_index,
        "target_start": target_interval[0],
        "target_stop": target_interval[1],
    }
    rows: list[dict[str, Any]] = []
    for field in ("Ne", "Pe", "Pi"):
        rows.append(
            {
                **base,
                "metric": "shared_state_standardized_rmse",
                "component": field,
                "band": "",
                "value": float(
                    physics["field_metrics_standardized"][field]["rmse"]
                ),
            }
        )
    for field in COMMON_FIELDS:
        for band in BAND_BOUNDS:
            ratio = float(physics["field_band_summaries"][field][band]["power_ratio"])
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(f"invalid spectral-power ratio for {field} {band}")
            rows.append(
                {
                    **base,
                    "metric": "spectral_power_absolute_log_ratio",
                    "component": field,
                    "band": band,
                    "value": abs(math.log(ratio)),
                }
            )
    for first, second in COMMON_CROSS_PAIRS:
        pair = f"{first}-{second}"
        for band in BAND_BOUNDS:
            rows.append(
                {
                    **base,
                    "metric": "complex_cross_spectrum_relative_l2",
                    "component": pair,
                    "band": band,
                    "value": complex_cross_band_relative_l2(physics, pair, band),
                }
            )
    for quantity in TRANSPORT_QUANTITIES:
        rows.append(
            {
                **base,
                "metric": "separatrix_transport_relative_l2",
                "component": quantity,
                "band": "",
                "value": float(
                    transport["quantities"][quantity]["separatrix"]["metrics"][
                        "relative_l2"
                    ]
                ),
            }
        )
    return rows


def decide(primary: list[Mapping[str, Any]], *, causal_phi_passed: bool) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    all_finite = True
    for family in FAMILIES:
        summaries = {}
        for metric, expected_count in EXPECTED_PRIMARY_COUNTS.items():
            values = np.asarray(
                [
                    float(row["value"])
                    for row in primary
                    if row["family"] == family and row["metric"] == metric
                ],
                dtype=np.float64,
            )
            if values.size != expected_count:
                raise ValueError(
                    f"{family} {metric} count {values.size} != {expected_count}"
                )
            finite = bool(np.all(np.isfinite(values)))
            all_finite &= finite
            summaries[metric] = {
                "count": int(values.size),
                "all_finite": finite,
                "median": float(np.median(values)) if finite else None,
                "minimum": float(np.min(values)) if finite else None,
                "maximum": float(np.max(values)) if finite else None,
            }
        by_family[family] = summaries
    c5p = by_family["c5p"]
    e6b = by_family["e6b"]
    ratios = {
        metric: float(e6b[metric]["median"] / c5p[metric]["median"])
        for metric in EXPECTED_PRIMARY_COUNTS
    }
    conditions = {
        "separatrix_transport_at_least_10_percent_better": (
            ratios["separatrix_transport_relative_l2"] <= 0.90
        ),
        "complex_cross_spectrum_strictly_better": (
            ratios["complex_cross_spectrum_relative_l2"] < 1.0
        ),
        "shared_state_not_worse_by_more_than_10_percent": (
            ratios["shared_state_standardized_rmse"] <= 1.10
        ),
        "spectral_power_not_worse_by_more_than_10_percent": (
            ratios["spectral_power_absolute_log_ratio"] <= 1.10
        ),
        "all_primary_scalars_finite": all_finite,
        "causal_exact_phi_provenance_passed": causal_phi_passed,
    }
    favor = all(conditions.values())
    return {
        "primary_summaries": by_family,
        "e6b_over_c5p_median_ratios": ratios,
        "conditions": conditions,
        "favor_e6b_saved_state": favor,
        "three_seed_confirmation_authorized": favor,
        "next_action": (
            "confirm_same_matched_pair_at_three_seeds"
            if favor
            else "retain_c5p_control_and_stop_saved_state_branch"
        ),
    }


def _update_accumulators(
    *,
    overall: PairAccumulator,
    block: PairAccumulator,
    truth_standardized: np.ndarray,
    candidate_standardized: np.ndarray,
    truth_physical: np.ndarray,
    candidate_physical: np.ndarray,
    truth_transport: Mapping[str, Mapping[str, np.ndarray]],
    candidate_transport: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    arguments = {
        "truth_standardized": truth_standardized,
        "candidate_standardized": candidate_standardized,
        "truth_physical": truth_physical,
        "candidate_physical": candidate_physical,
        "truth_transport": truth_transport,
        "candidate_transport": candidate_transport,
    }
    overall.update(**arguments)
    block.update(**arguments)


def score_in_memory_candidate(
    *,
    target_frames: np.ndarray,
    blocks: list[list[int]],
    truth_standardized: np.ndarray,
    truth_physical: np.ndarray,
    truth_transport: Mapping[str, Mapping[str, np.ndarray]],
    candidate_standardized: np.ndarray,
    candidate_physical: np.ndarray,
    candidate_transport: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    overall = PairAccumulator()
    block_results = []
    for block_index, (target_start, target_stop) in enumerate(blocks, start=1):
        positions = np.flatnonzero(
            (target_frames >= target_start) & (target_frames < target_stop)
        )
        if positions.size != target_stop - target_start or np.any(np.diff(positions) != 1):
            raise ValueError("persistence block coverage differs")
        block = PairAccumulator()
        for start in range(int(positions[0]), int(positions[-1]) + 1, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, int(positions[-1]) + 1)
            _update_accumulators(
                overall=overall,
                block=block,
                truth_standardized=truth_standardized[start:stop],
                candidate_standardized=candidate_standardized[start:stop],
                truth_physical=truth_physical[start:stop],
                candidate_physical=candidate_physical[start:stop],
                truth_transport=_transport_slice(truth_transport, start, stop),
                candidate_transport=_transport_slice(candidate_transport, start, stop),
            )
        block_results.append(
            {
                "block_index": block_index,
                "target_frames": [target_start, target_stop],
                "metrics": block.finalize(),
            }
        )
    return {"overall": overall.finalize(), "blocks": block_results}


def score_forecast_method(
    *,
    family: str,
    horizon: int,
    method: str,
    blocks: list[list[int]],
    forecast: h5py.File,
    exact_index: Mapping[str, Mapping[str, Path]],
    normalization: Any,
    target_frames: np.ndarray,
    truth_standardized: np.ndarray,
    truth_physical: np.ndarray,
    truth_transport: Mapping[str, Mapping[str, np.ndarray]],
    geometry: Any,
    examples: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float | int]]:
    group = forecast[f"horizon_{horizon}/methods/{method}"]
    overall = PairAccumulator()
    block_results = []
    surface = {
        quantity: np.empty(target_frames.size, dtype=np.float64)
        for quantity in TRANSPORT_QUANTITIES
    }
    density = {
        "cell_count": 0,
        "nonpositive_density_count": 0,
        "below_density_floor_count": 0,
        "minimum_density": math.inf,
        "maximum_absolute_derived_Vi": 0.0,
    }
    candidate_name = f"h{horizon}_{method}_predicted_e6b_native81.h5"
    with ExitStack() as stack:
        candidate_handle = None
        phi_handle = None
        if family == "e6b":
            if candidate_name not in exact_index:
                raise ValueError(f"missing exact phi for {candidate_name}")
            pair = exact_index[candidate_name]
            candidate_handle = stack.enter_context(h5py.File(pair["candidate"], "r"))
            phi_handle = stack.enter_context(h5py.File(pair["phi"], "r"))
            candidate_frames = np.asarray(
                candidate_handle["coordinates/frame_index"][:], dtype=np.int64
            )
            phi_frames = np.asarray(phi_handle["frame_index"][:], dtype=np.int64)
            if (
                not np.array_equal(candidate_frames, target_frames)
                or not np.array_equal(phi_frames, target_frames)
                or bool(candidate_handle.attrs["target_truth_used_during_generation"])
                or candidate_handle.attrs["boundary_policy"]
                != "predicted_Bphi_no_truth_bypass"
                or bool(phi_handle.attrs["truth_layout"])
            ):
                raise ValueError("E6B candidate/exact-phi coordinates differ")

        for block_index, (target_start, target_stop) in enumerate(blocks, start=1):
            positions = np.flatnonzero(
                (target_frames >= target_start) & (target_frames < target_stop)
            )
            if positions.size != target_stop - target_start or np.any(np.diff(positions) != 1):
                raise ValueError("forecast block coverage differs")
            block = PairAccumulator()
            for start in range(int(positions[0]), int(positions[-1]) + 1, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, int(positions[-1]) + 1)
                predicted_state = np.asarray(group["volume"][start:stop], dtype=np.float32)
                if not np.all(np.isfinite(predicted_state)):
                    raise ValueError(f"{family} forecast state is non-finite")
                if family == "c5p":
                    candidate_standardized = predicted_state
                    candidate_physical = decode_physical_batch(
                        normalization, FAMILY_FIELDS["c5p"], predicted_state
                    )
                    candidate_transport = transport_from_model88(
                        candidate_physical, geometry
                    )
                else:
                    assert candidate_handle is not None and phi_handle is not None
                    evolved_physical = decode_physical_batch(
                        normalization, FAMILY_FIELDS["e6b"], predicted_state
                    )
                    phi_native = np.asarray(phi_handle["phi"][start:stop], dtype=np.float64)
                    phi_model88 = periodic_resample_float32(
                        np.asarray(phi_native, dtype=np.float32), 88, axis=-1
                    ).astype(np.float64)
                    candidate_physical, diagnostics = e6b_common_physical(
                        evolved_physical, phi_model88
                    )
                    candidate_standardized = encode_physical_batch(
                        normalization, COMMON_FIELDS, candidate_physical
                    )
                    candidate_transport = transport_from_native(
                        ne=np.asarray(candidate_handle["candidate/Ne"][start:stop]),
                        pe=np.asarray(candidate_handle["candidate/Pe"][start:stop]),
                        pi=np.asarray(candidate_handle["candidate/Pi"][start:stop]),
                        phi=phi_native,
                        geometry=geometry,
                    )
                    density["cell_count"] += int(diagnostics["cell_count"])
                    density["nonpositive_density_count"] += int(
                        diagnostics["nonpositive_density_count"]
                    )
                    density["below_density_floor_count"] += int(
                        diagnostics["below_density_floor_count"]
                    )
                    density["minimum_density"] = min(
                        float(density["minimum_density"]),
                        float(diagnostics["minimum_density"]),
                    )
                    density["maximum_absolute_derived_Vi"] = max(
                        float(density["maximum_absolute_derived_Vi"]),
                        float(diagnostics["maximum_absolute_derived_Vi"]),
                    )
                _update_accumulators(
                    overall=overall,
                    block=block,
                    truth_standardized=truth_standardized[start:stop],
                    candidate_standardized=candidate_standardized,
                    truth_physical=truth_physical[start:stop],
                    candidate_physical=candidate_physical,
                    truth_transport=_transport_slice(truth_transport, start, stop),
                    candidate_transport=candidate_transport,
                )
                for quantity in TRANSPORT_QUANTITIES:
                    surface[quantity][start:stop] = candidate_transport[quantity][
                        "separatrix_wedge"
                    ]
                positions_560 = np.flatnonzero(target_frames[start:stop] == 560)
                if positions_560.size:
                    examples[
                        f"h{horizon}_{family}_{method}_target560"
                    ] = np.asarray(
                        candidate_physical[int(positions_560[0])], dtype=np.float32
                    )
            block_results.append(
                {
                    "block_index": block_index,
                    "target_frames": [target_start, target_stop],
                    "metrics": block.finalize(),
                }
            )
    if family == "c5p":
        density = {
            "cell_count": int(target_frames.size * np.prod(VOLUME_SHAPE)),
            "nonpositive_density_count": int(
                overall.matched.nonpositive_density_count
            ),
            "below_density_floor_count": None,
            "minimum_density": float(overall.matched.minimum_reconstructed_density),
            "maximum_absolute_derived_Vi": None,
        }
    return (
        {"overall": overall.finalize(), "blocks": block_results},
        surface,
        density,
    )


def main() -> None:
    args = parse_args()
    for path in (args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
        if "85606" in str(path).lower():
            raise ValueError("held-out 85606 paths are prohibited")
    if args.output.exists():
        raise FileExistsError(args.output)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 checkout commit differs")
    manifest = authorize_manifest(args.manifest, args.manifest_sha256)
    args.output.mkdir(parents=True)

    dependencies = manifest["dependencies"]
    artifact_root = Path(
        str(dependencies["model_dataset_manifest"]["path"])
    ).parent
    catalog = load_official_catalog(artifact_root)
    validation_states = load_validation_states(catalog)
    native_truth = NativeTruthCatalog(
        load_strict_json(Path(dependencies["native_truth_result"]["path"]))
    )
    geometry = load_transport_geometry(
        geometry_path=Path(dependencies["geometry"]["path"]),
        geometry_manifest=load_strict_json(
            Path(dependencies["geometry_manifest"]["path"])
        ),
    )
    exact_index = exact_phi_index(manifest)

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
        job_type="old-85604-matched-state-bounded-physics",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "matched-state-view",
            "bounded-rollout",
            "exact-phi",
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
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "paper0_commit": args.paper0_commit,
            "manifest_sha256": args.manifest_sha256,
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
    primary: list[dict[str, Any]] = []
    physics: dict[str, Any] = {family: {} for family in FAMILIES}
    persistence: dict[str, Any] = {}
    examples: dict[str, np.ndarray] = {}
    surface_series: dict[str, np.ndarray] = {}
    density_diagnostics: dict[str, Any] = {family: {} for family in FAMILIES}
    try:
        handles = {}
        with ExitStack() as stack:
            for family in FAMILIES:
                forecast_path = Path(
                    manifest["generation_results"][family]["forecast"]["path"]
                )
                handle = stack.enter_context(h5py.File(forecast_path, "r"))
                validate_forecast_schema(handle, family=family)
                handles[family] = handle

            for horizon in HORIZONS:
                target_frames = np.arange(
                    VALIDATION_START + horizon, VALIDATION_STOP, dtype=np.int64
                )
                current_frames = target_frames - horizon
                blocks = manifest["evaluation"]["target_frame_blocks"][str(horizon)]
                truth_standardized = np.asarray(
                    validation_states[target_frames - VALIDATION_START],
                    dtype=np.float32,
                )
                current_standardized = np.asarray(
                    validation_states[current_frames - VALIDATION_START],
                    dtype=np.float32,
                )
                truth_physical = decode_physical_batch(
                    catalog.normalization, COMMON_FIELDS, truth_standardized
                )
                current_physical = decode_physical_batch(
                    catalog.normalization, COMMON_FIELDS, current_standardized
                )
                truth_native = native_truth.read(
                    int(target_frames[0]), VALIDATION_STOP,
                    fields=("Ne", "Pe", "Pi", "phi"),
                )
                current_native = native_truth.read(
                    int(current_frames[0]), int(current_frames[-1]) + 1,
                    fields=("Ne", "Pe", "Pi", "phi"),
                )
                truth_transport = transport_from_native(
                    ne=truth_native["Ne"],
                    pe=truth_native["Pe"],
                    pi=truth_native["Pi"],
                    phi=truth_native["phi"],
                    geometry=geometry,
                )
                current_transport = transport_from_native(
                    ne=current_native["Ne"],
                    pe=current_native["Pe"],
                    pi=current_native["Pi"],
                    phi=current_native["phi"],
                    geometry=geometry,
                )
                persistence[str(horizon)] = score_in_memory_candidate(
                    target_frames=target_frames,
                    blocks=blocks,
                    truth_standardized=truth_standardized,
                    truth_physical=truth_physical,
                    truth_transport=truth_transport,
                    candidate_standardized=current_standardized,
                    candidate_physical=current_physical,
                    candidate_transport=current_transport,
                )
                examples[f"h{horizon}_truth_target560"] = np.asarray(
                    truth_physical[int(np.flatnonzero(target_frames == 560)[0])],
                    dtype=np.float32,
                )
                examples[f"h{horizon}_persistence_target560"] = np.asarray(
                    current_physical[int(np.flatnonzero(target_frames == 560)[0])],
                    dtype=np.float32,
                )
                for quantity in TRANSPORT_QUANTITIES:
                    surface_series[f"h{horizon}_truth_{quantity}"] = np.asarray(
                        truth_transport[quantity]["separatrix_wedge"], dtype=np.float64
                    )
                    surface_series[f"h{horizon}_persistence_{quantity}"] = np.asarray(
                        current_transport[quantity]["separatrix_wedge"], dtype=np.float64
                    )
                surface_series[f"h{horizon}_target_frame"] = target_frames

                for family in FAMILIES:
                    physics[family][str(horizon)] = {}
                    density_diagnostics[family][str(horizon)] = {}
                    for method in method_schedule(horizon):
                        scored, method_surface, density = score_forecast_method(
                            family=family,
                            horizon=horizon,
                            method=method,
                            blocks=blocks,
                            forecast=handles[family],
                            exact_index=exact_index,
                            normalization=catalog.normalization,
                            target_frames=target_frames,
                            truth_standardized=truth_standardized,
                            truth_physical=truth_physical,
                            truth_transport=truth_transport,
                            geometry=geometry,
                            examples=examples,
                        )
                        physics[family][str(horizon)][method] = scored
                        density_diagnostics[family][str(horizon)][method] = density
                        for block in scored["blocks"]:
                            primary.extend(
                                primary_rows(
                                    family=family,
                                    horizon=horizon,
                                    method=method,
                                    block_index=int(block["block_index"]),
                                    target_interval=tuple(block["target_frames"]),
                                    metrics=block["metrics"],
                                )
                            )
                        for quantity, values in method_surface.items():
                            surface_series[
                                f"h{horizon}_{family}_{method}_{quantity}"
                            ] = values

        decision = decide(primary, causal_phi_passed=True)
        run.log(
            {
                "decision/favor_e6b_saved_state": int(
                    decision["favor_e6b_saved_state"]
                ),
                **{
                    f"decision/e6b_over_c5p/{name}": value
                    for name, value in decision[
                        "e6b_over_c5p_median_ratios"
                    ].items()
                },
            },
            step=0,
        )
        metrics_record = {
            "schema_version": 1,
            "scope": SCOPE,
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "physics_derived_loss_used": False,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "common_fields": list(COMMON_FIELDS),
            "persistence": _json_safe(persistence),
            "by_state_view": _json_safe(physics),
            "density_diagnostics": _json_safe(density_diagnostics),
            "decision": _json_safe(decision),
        }
        metrics_path = args.output / "physics_metrics.json"
        atomic_json(metrics_path, metrics_record)
        primary_path = args.output / "primary_metrics.csv"
        with primary_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(primary[0]))
            writer.writeheader()
            writer.writerows(primary)
        examples_path = args.output / "example_common_physical_target560.npz"
        np.savez_compressed(examples_path, **examples)
        surface_path = args.output / "separatrix_transport_series.npz"
        np.savez_compressed(surface_path, **surface_series)

        result = {
            "schema_version": 1,
            "scope": SCOPE,
            "status": "paired_bounded_physics_scored",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "physics_derived_loss_used": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "steering_performed": False,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": {
                "path": str(args.manifest),
                "sha256": args.manifest_sha256,
            },
            "physics_metrics": {
                "path": str(metrics_path),
                "sha256": sha256_path(metrics_path),
            },
            "primary_metrics": {
                "path": str(primary_path),
                "sha256": sha256_path(primary_path),
                "row_count": len(primary),
            },
            "examples": {
                "path": str(examples_path),
                "sha256": sha256_path(examples_path),
            },
            "separatrix_transport_series": {
                "path": str(surface_path),
                "sha256": sha256_path(surface_path),
            },
            "decision": decision,
            "wall_seconds_before_wandb_verification": time.perf_counter()
            - started,
        }
        run.summary.update(
            {
                "final/status": result["status"],
                "final/favor_e6b_saved_state": decision[
                    "favor_e6b_saved_state"
                ],
                "final/next_action": decision["next_action"],
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
    atomic_json(
        args.output / "wandb.json",
        {
            "schema_version": 1,
            "required": True,
            "mode": "online",
            "spec": spec.to_record(),
            "authenticated_username": str(getattr(viewer, "username", "")),
            "wandb_version": wandb.__version__,
            "run_url": run_url,
            "remote_path": remote_path,
            "remote_state_after_finish": remote_state,
            "local_artifacts_are_scientific_authority": True,
        },
    )
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
