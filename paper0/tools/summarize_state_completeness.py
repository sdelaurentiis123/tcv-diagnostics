#!/usr/bin/env python3
"""Compact a completed state-completeness audit without changing its metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_FIELDS = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "Ve", "Vi")
EXPECTED_RELATIONS = (
    "NVe_from_softfloor_Ne_Ve",
    "NVi_from_softfloor_Ne_Vi",
    "NVe_from_plain_Ne_Ve",
    "NVi_from_plain_Ne_Vi",
)
SCOPES = (
    "full_physical_domain",
    "guard_independent_transport_interior",
    "target_dependent_rows",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is prohibited: {value}")


def strict_json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def strict_json_write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary file {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def compact_field(statistics: dict[str, Any]) -> dict[str, Any]:
    return {
        scope: {
            key: statistics["scopes"][scope][key]
            for key in (
                "total_count",
                "finite_count",
                "nonfinite_count",
                "rms",
                "minimum",
                "maximum",
            )
        }
        for scope in SCOPES
    }


def compact_closure(statistics: dict[str, Any]) -> dict[str, Any]:
    return {
        "scopes": {
            scope: {
                key: statistics["scopes"][scope][key]
                for key in (
                    "total_count",
                    "nonfinite_count",
                    "point_discrepancy_count",
                    "relative_l2_error",
                    "maximum_error",
                    "frame_pass_count",
                    "frame_fail_count",
                    "failed_frame_indices",
                )
            }
            for scope in SCOPES
        },
        "point_discrepancy_count_by_temporal_block": statistics[
            "point_discrepancy_count_by_temporal_block"
        ],
    }


def validate_raw(raw: dict[str, Any]) -> None:
    if raw.get("phase") != "phase2_85604_state_completeness_audit":
        raise ValueError("input is not a complete state-completeness audit")
    if not raw.get("audit_completed") or not raw.get("rank_shard_completed"):
        raise ValueError("input audit is incomplete")
    if raw.get("development_run") != "85604" or raw.get("held_out_85606_read"):
        raise ValueError("input violates the 85604-only scope")
    if raw.get("rank_file_count") != 256 or raw.get("rank_indices") != list(range(256)):
        raise ValueError("input lacks exact all-rank coverage")
    if raw.get("shape_per_field") != [624, 64, 32, 81]:
        raise ValueError("input canonical shape differs from the frozen protocol")
    if raw.get("zperiod") != 5:
        raise ValueError("input zperiod differs from the frozen protocol")
    if set(raw.get("field_statistics", {})) != set(EXPECTED_FIELDS):
        raise ValueError("input field statistics are incomplete")
    relations = raw.get("closure_statistics", {}).get("relations", {})
    if set(relations) != set(EXPECTED_RELATIONS):
        raise ValueError("input closure relations are incomplete")
    metadata = raw.get("variable_metadata", {})
    if len(metadata) != 11:
        raise ValueError("input metadata inventory does not contain eleven fields")
    digests = raw.get("guard_stripped_stream_digest_tree", {})
    if set(digests) != set(EXPECTED_FIELDS):
        raise ValueError("input stream digest tree is incomplete")
    if any(SHA256_PATTERN.fullmatch(value) is None for value in digests.values()):
        raise ValueError("input contains a malformed stream digest")
    if SHA256_PATTERN.fullmatch(raw.get("paper0_commit", "")) is not None:
        raise ValueError("Git commits are SHA-1 here, not SHA-256")
    if re.fullmatch(r"[0-9a-f]{40}", raw.get("paper0_commit", "")) is None:
        raise ValueError("input executed commit is malformed")


def compact_result(
    raw: dict[str, Any],
    *,
    raw_path: str,
    raw_sha256: str,
    compactor_commit: str,
) -> dict[str, Any]:
    validate_raw(raw)
    if SHA256_PATTERN.fullmatch(raw_sha256) is None:
        raise ValueError("raw artifact SHA-256 is malformed")
    if re.fullmatch(r"[0-9a-f]{40}", compactor_commit) is None:
        raise ValueError("compactor commit is malformed")
    floor = raw["density_floor_statistics"]
    closures = raw["closure_statistics"]
    return {
        "schema_version": 1,
        "phase": "phase2_85604_state_completeness_compact",
        "development_run": "85604",
        "held_out_85606_read": False,
        "raw_artifact": {
            "path": raw_path,
            "sha256": raw_sha256,
            "slurm_job_id": raw["slurm_job_id"],
            "executed_paper0_commit": raw["paper0_commit"],
            "manifest": raw["manifest"],
            "manifest_sha256": raw["manifest_sha256"],
            "rank_shard_digest_tree_sha256": raw[
                "rank_shard_digest_tree_sha256"
            ],
        },
        "compactor_commit": compactor_commit,
        "coverage": {
            "rank_file_count": raw["rank_file_count"],
            "frame_count": raw["frame_count"],
            "shape_per_field": raw["shape_per_field"],
            "total_points_per_stream": raw["total_points_per_stream"],
            "processor_coverage": raw["processor_coverage"],
            "zperiod": raw["zperiod"],
            "native_z_samples": raw["native_z_samples"],
            "normalized_time": raw["normalized_time"],
            "metadata_inventory_field_count": len(raw["variable_metadata"]),
            "stream_digest_tree": raw["guard_stripped_stream_digest_tree"],
        },
        "field_statistics": {
            field: compact_field(raw["field_statistics"][field])
            for field in EXPECTED_FIELDS
        },
        "density_floor_statistics": {
            "density_floor": floor["density_floor"],
            "scope_counts": floor["scope_counts"],
            "count_by_temporal_block": floor["count_by_temporal_block"],
        },
        "closure_statistics": {
            "atol": closures["atol"],
            "rtol": closures["rtol"],
            "relations": {
                relation: compact_closure(closures["relations"][relation])
                for relation in EXPECTED_RELATIONS
            },
        },
        "scientific_findings": raw["scientific_findings"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compactor-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = strict_json_load(args.input)
    compact = compact_result(
        raw,
        raw_path=str(args.input),
        raw_sha256=sha256_file(args.input),
        compactor_commit=args.compactor_commit,
    )
    strict_json_write(args.output, compact)
    print(json.dumps(compact["scientific_findings"], indent=2, sort_keys=True))
    print(f"Wrote compact result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
