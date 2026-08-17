#!/usr/bin/env python3
"""Create a compact, figure-complete record of the O1 transport oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


CODECS = ("f8", "z44")
QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)
COMPARISONS = (
    "P0_vs_P1_state_gap",
    "P1_vs_P2_input_roundtrip",
    "P2_vs_R_codec_only",
    "P0_vs_R_authoritative",
)


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


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.dtype.str.encode("utf-8"))
    digest.update(array.tobytes())
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
            json.dump(
                _json_safe(record),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    if set(summary["comparisons"]) != set(COMPARISONS):
        raise ValueError("transport comparison set changed")
    comparisons: dict[str, Any] = {}
    for comparison in COMPARISONS:
        source = summary["comparisons"][comparison]
        if set(source["quantities"]) != set(QUANTITIES):
            raise ValueError(f"quantity set changed for {comparison}")
        comparisons[comparison] = {
            "reference_path": source["reference_path"],
            "candidate_path": source["candidate_path"],
            "quantities": source["quantities"],
        }
    return {"frames": summary["frames"], "comparisons": comparisons}


def _compact_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_index": block["block_index"],
        "start_inclusive": block["start_inclusive"],
        "stop_exclusive": block["stop_exclusive"],
        "metrics": _compact_summary(block["metrics"]),
    }


def _surface_statistics(
    truth: np.ndarray,
    reconstruction: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    if truth.shape != (624,) or reconstruction.shape != truth.shape:
        raise ValueError("surface series must contain exactly 624 paired frames")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(reconstruction)):
        raise ValueError("surface series contains non-finite values")
    return {
        "truth_mean": float(np.mean(truth)),
        "truth_std": float(np.std(truth, ddof=0)),
        "truth_minimum": float(np.min(truth)),
        "truth_maximum": float(np.max(truth)),
        "reconstruction_mean": float(np.mean(reconstruction)),
        "reconstruction_std": float(np.std(reconstruction, ddof=0)),
        "reconstruction_minimum": float(np.min(reconstruction)),
        "reconstruction_maximum": float(np.max(reconstruction)),
        "rmse": float(np.sqrt(np.mean(np.square(reconstruction - truth)))),
    }


def compact_record(
    raw: dict[str, Any],
    *,
    artifact_path: str,
    digest: str,
) -> dict[str, Any]:
    codec_results: dict[str, Any] = {}
    shared_truth_si: dict[str, Any] | None = None
    surface_reconstructions_si: dict[str, Any] = {}
    surface_statistics: dict[str, Any] = {}
    surface_digests: dict[str, Any] = {}

    for codec_name in CODECS:
        source = raw["codec_results"][codec_name]
        overall = source["overall"]
        compact_overall = _compact_summary(overall)
        truth_si = {
            quantity: overall["surface_series_si"]["P0"][quantity]
            for quantity in QUANTITIES
        }
        reconstruction_si = {
            quantity: overall["surface_series_si"]["R"][quantity]
            for quantity in QUANTITIES
        }
        if shared_truth_si is None:
            shared_truth_si = truth_si
        else:
            for quantity in QUANTITIES:
                np.testing.assert_array_equal(
                    np.asarray(shared_truth_si[quantity], dtype=np.float64),
                    np.asarray(truth_si[quantity], dtype=np.float64),
                )
        surface_reconstructions_si[codec_name] = reconstruction_si
        surface_statistics[codec_name] = {
            quantity: _surface_statistics(
                np.asarray(truth_si[quantity], dtype=np.float64),
                np.asarray(reconstruction_si[quantity], dtype=np.float64),
            )
            for quantity in QUANTITIES
        }
        surface_digests[codec_name] = {
            "truth_P0": {
                quantity: _array_sha256(truth_si[quantity])
                for quantity in QUANTITIES
            },
            "reconstruction_R": {
                quantity: _array_sha256(reconstruction_si[quantity])
                for quantity in QUANTITIES
            },
        }
        codec_results[codec_name] = {
            "identity": source["identity"],
            "parameter_count": source["parameter_count"],
            "nonpositive_reconstruction": source["nonpositive_reconstruction"],
            "overall": compact_overall,
            "temporal_blocks": [
                _compact_block(block) for block in source["temporal_blocks"]
            ],
            "gate": source["gate"],
            "shared_truth_path_sha256": source["shared_truth_path_sha256"],
        }

    if shared_truth_si is None:
        raise ValueError("codec results are empty")
    alignment_series = raw["alignment"]["per_frame_relative_l2"]
    if set(alignment_series) != {"Ne", "Te", "Ti", "phi", "Vi"}:
        raise ValueError("input alignment field set changed")
    alignment_digests = {
        field: _array_sha256(values) for field, values in alignment_series.items()
    }
    return {
        "schema_version": "0.1.0",
        "result_type": "phase2_o1_codec_transport_oracle_compact",
        "status": raw["status"],
        "scope": raw["scope"],
        "execution": raw["execution"],
        "provenance": raw["provenance"],
        "raw_artifact": {
            "path": artifact_path,
            "sha256": digest,
            "tracked_in_git": False,
        },
        "alignment": {
            "time_exact": raw["alignment"]["time_exact"],
            "x_coordinates_exact": raw["alignment"]["x_coordinates_exact"],
            "y_coordinates_exact": raw["alignment"]["y_coordinates_exact"],
            "maximum_per_frame_relative_l2": raw["alignment"][
                "maximum_per_frame_relative_l2"
            ],
            "per_frame_series_sha256": alignment_digests,
        },
        "geometry": raw["geometry"],
        "units": raw["units"],
        "shared_truth": raw["shared_truth"],
        "surface_series_si": {
            "units": raw["codec_results"]["f8"]["overall"][
                "surface_series_si_units"
            ],
            "truth_P0": shared_truth_si,
            "reconstruction_R": surface_reconstructions_si,
            "series_sha256": surface_digests,
        },
        "surface_statistics_si": surface_statistics,
        "codec_results": codec_results,
        "decision": raw["decision"],
    }


def validate_raw(
    raw: dict[str, Any],
    *,
    expected_commit: str,
    expected_job: str,
) -> None:
    if raw.get("result_type") != "phase2_o1_codec_transport_oracle":
        raise ValueError(f"unexpected result type {raw.get('result_type')}")
    if raw.get("status") != "completed":
        raise ValueError("O1 transport result is not complete")
    scope = raw["scope"]
    if (
        scope.get("run_id") != "85604"
        or scope.get("shot_85606_accessed") is not False
        or scope.get("training_performed") is not False
        or scope.get("frame_count") != 624
    ):
        raise ValueError("O1 transport source violates the frozen scope")
    if raw["execution"].get("paper0_commit") != expected_commit:
        raise ValueError("O1 transport source commit differs from expectation")
    if str(raw["execution"].get("slurm_job_id")) != str(expected_job):
        raise ValueError("O1 transport source job differs from expectation")
    if set(raw["codec_results"]) != set(CODECS):
        raise ValueError("codec result set changed")
    if not (
        raw["shared_truth"]["bitwise_digest_identical"]
        and raw["shared_truth"]["single_evaluation_fed_to_both_codec_comparisons"]
    ):
        raise ValueError("truth paths are not certified identical")
    truth_hashes = {
        raw["codec_results"][codec]["shared_truth_path_sha256"]
        for codec in CODECS
    }
    if truth_hashes != {raw["shared_truth"]["sha256"]}:
        raise ValueError("codec truth-path hashes disagree")


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
    validate_raw(
        raw,
        expected_commit=args.expected_commit,
        expected_job=args.expected_job,
    )
    record = compact_record(raw, artifact_path=args.artifact_path, digest=actual_digest)
    _write_json_atomic(output_path, record)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
