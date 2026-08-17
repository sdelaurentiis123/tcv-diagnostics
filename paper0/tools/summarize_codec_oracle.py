#!/usr/bin/env python3
"""Create a compact, figure-complete record from the full O1 metric artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = ("Ne", "Te", "Ti", "phi", "Vi")
PRIMARY_PAIRS = ("Ne-phi", "Te-phi", "Ti-phi")
PRIMARY_BANDS = (
    "low_nonaxisymmetric",
    "coherent_study",
    "upper_study",
    "measured_high",
    "remaining_resolved",
)
MAX_MODE_K = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-job", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _mode_curves(overall: dict[str, Any]) -> dict[str, Any]:
    stored_k = overall["stored_k"][: MAX_MODE_K + 1]
    full_torus_n = overall["full_torus_n"][: MAX_MODE_K + 1]
    curves: dict[str, Any] = {
        "stored_k": stored_k,
        "full_torus_n": full_torus_n,
        "fields": {},
    }
    for field in FIELDS:
        source = overall["toroidal_spectral_curves_linear_coordinates"][field]
        truth_power = np.asarray(source["truth_power"], dtype=np.float64)
        nonaxisymmetric_total = float(np.sum(truth_power[1:]))
        fraction = truth_power / nonaxisymmetric_total
        curves["fields"][field] = {
            "truth_power": truth_power[: MAX_MODE_K + 1].tolist(),
            "truth_power_fraction_of_nonaxisymmetric": fraction[
                : MAX_MODE_K + 1
            ].tolist(),
            "reconstruction_power_ratio": source["power_ratio"][
                : MAX_MODE_K + 1
            ],
            "truth_to_reconstruction_coherence": source[
                "truth_to_reconstruction_coherence"
            ][: MAX_MODE_K + 1],
            "truth_to_reconstruction_phase_radians": source[
                "truth_to_reconstruction_phase_radians"
            ][: MAX_MODE_K + 1],
        }
    return curves


def _cross_mode_curves(overall: dict[str, Any]) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for pair in PRIMARY_PAIRS:
        source = overall["cross_field_curves_linear_coordinates"][pair]
        truth_coherence = np.asarray(source["truth_coherence"], dtype=np.float64)
        reconstruction_coherence = np.asarray(
            source["reconstruction_coherence"], dtype=np.float64
        )
        curves[pair] = {
            "truth_coherence": truth_coherence[: MAX_MODE_K + 1].tolist(),
            "reconstruction_coherence": reconstruction_coherence[
                : MAX_MODE_K + 1
            ].tolist(),
            "absolute_coherence_change": np.abs(
                reconstruction_coherence - truth_coherence
            )[: MAX_MODE_K + 1].tolist(),
            "truth_phase_radians": source["truth_phase_radians"][
                : MAX_MODE_K + 1
            ],
            "reconstruction_phase_radians": source[
                "reconstruction_phase_radians"
            ][: MAX_MODE_K + 1],
            "signed_phase_error_radians": source["signed_phase_error_radians"][
                : MAX_MODE_K + 1
            ],
        }
    return curves


def _compact_block(block: dict[str, Any]) -> dict[str, Any]:
    metrics = block["metrics"]
    return {
        "block_index": block["block_index"],
        "start_inclusive": block["start_inclusive"],
        "stop_exclusive": block["stop_exclusive"],
        "aggregate_five_field_rmse_legacy_standardized": metrics[
            "aggregate_five_field_rmse_legacy_standardized"
        ],
        "field_metrics_legacy_standardized": metrics[
            "field_metrics_legacy_standardized"
        ],
        "phi_gauge_fixed_metrics_legacy_standardized": metrics[
            "phi_gauge_fixed_metrics_legacy_standardized"
        ],
        "field_band_summaries": {
            field: {
                band: metrics["field_band_summaries"][field][band]
                for band in PRIMARY_BANDS
            }
            for field in FIELDS
        },
        "cross_field_band_summaries": {
            pair: {
                band: metrics["cross_field_band_summaries"][pair][band]
                for band in PRIMARY_BANDS
            }
            for pair in PRIMARY_PAIRS
        },
    }


def compact_record(raw: dict[str, Any], *, artifact_path: str, digest: str) -> dict[str, Any]:
    codecs: dict[str, Any] = {}
    for codec_name in ("f8", "z44"):
        source = raw["codec_results"][codec_name]
        overall = source["overall"]
        codecs[codec_name] = {
            "identity": source["identity"],
            "parameter_count": source["parameter_count"],
            "chunk_frames": source["chunk_frames"],
            "elapsed_seconds": source["elapsed_seconds"],
            "peak_cuda_memory_bytes": source["peak_cuda_memory_bytes"],
            "preliminary_gate": source["preliminary_gate"],
            "aggregate_five_field_rmse_legacy_standardized": overall[
                "aggregate_five_field_rmse_legacy_standardized"
            ],
            "field_metrics_legacy_standardized": overall[
                "field_metrics_legacy_standardized"
            ],
            "phi_gauge_fixed_metrics_legacy_standardized": overall[
                "phi_gauge_fixed_metrics_legacy_standardized"
            ],
            "density_linear_reconstruction": overall[
                "density_linear_reconstruction"
            ],
            "field_band_summaries": overall["field_band_summaries"],
            "cross_field_band_summaries": overall[
                "cross_field_band_summaries"
            ],
            "mode_curves_k0_to_k16": _mode_curves(overall),
            "cross_mode_curves_k0_to_k16": _cross_mode_curves(overall),
            "temporal_blocks": [
                _compact_block(block) for block in source["temporal_blocks"]
            ],
        }
    return {
        "schema_version": "0.1.0",
        "result_type": "phase2_o1_codec_reconstruction_oracle_compact",
        "status": raw["status"],
        "protocol_decision": raw["protocol_decision"],
        "scope": raw["scope"],
        "execution": raw["execution"],
        "data": raw["data"],
        "raw_artifact": {
            "path": artifact_path,
            "sha256": digest,
            "tracked_in_git": False,
        },
        "truth_metrics_identical_between_codec_passes": raw[
            "truth_metrics_identical_between_codec_passes"
        ],
        "codec_results": codecs,
        "claims": raw["claims"],
    }


def main() -> None:
    args = parse_args()
    source_path = Path(args.input).expanduser().resolve(strict=True)
    output_path = Path(args.output).expanduser().resolve(strict=False)
    actual_digest = sha256_file(source_path)
    if actual_digest != args.expected_sha256:
        raise ValueError(
            f"input SHA-256 {actual_digest} != expected {args.expected_sha256}"
        )
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if raw.get("result_type") != "phase2_o1_codec_reconstruction_oracle":
        raise ValueError(f"unexpected result type {raw.get('result_type')}")
    if raw.get("status") != "completed":
        raise ValueError(f"O1 result is not completed: {raw.get('status')}")
    if raw["scope"].get("run_id") != "85604":
        raise ValueError("O1 source is not run 85604")
    if raw["scope"].get("shot_85606_accessed") is not False:
        raise ValueError("O1 source does not certify 85606 exclusion")
    if raw["execution"].get("paper0_commit") != args.expected_commit:
        raise ValueError("O1 source commit does not match expectation")
    if str(raw["execution"].get("slurm_job_id")) != str(args.expected_job):
        raise ValueError("O1 SLURM job does not match expectation")
    record = compact_record(
        raw, artifact_path=args.artifact_path, digest=actual_digest
    )
    _write_json_atomic(output_path, record)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
