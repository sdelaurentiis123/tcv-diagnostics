#!/usr/bin/env python3
"""Complete one matched O1 codec result with common-view and transport gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.codec_transport import (  # noqa: E402
    build_matched_o1_transport_gate,
)
from tcv_diagnostics.matched_codec_metrics import (  # noqa: E402
    build_matched_o1_view_gate,
    training_materiality,
)
from tcv_diagnostics.matched_o1_evaluation import (  # noqa: E402
    evaluate_e6b_common_interval,
    evaluate_matched_transport_interval,
)
from tcv_diagnostics.matched_o1_transport import (  # noqa: E402
    MatchedCandidateArtifact,
    MatchedPhiArtifact,
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import (  # noqa: E402
    OFFICIAL_ARTIFACT_ROOT,
    TRAIN_INTERVAL,
    VALIDATION_INTERVAL,
    load_official_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-result", type=Path, required=True)
    parser.add_argument("--reconstruction-result-sha256", required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--train-phi-result", type=Path)
    parser.add_argument("--train-phi-result-sha256")
    parser.add_argument("--validation-phi-result", type=Path)
    parser.add_argument("--validation-phi-result-sha256")
    parser.add_argument("--truth-replay-summary", type=Path)
    parser.add_argument("--truth-replay-summary-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--chunk-frames", type=int, default=1)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"Paper 0 commit mismatch: {actual} != {expected_commit}")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def load_hash_locked_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    if sha256_path(resolved) != expected_sha256:
        raise ValueError(f"JSON SHA-256 differs: {resolved}")
    return load_strict_json(resolved)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("final matched O1 result contains a non-finite number")
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def validate_reconstruction_result(record: Mapping[str, Any]) -> dict[str, Any]:
    if (
        record.get("scope") != "phase2_matched_o1_codec_reconstruction"
        or record.get("status") != "completed_pending_exact_elliptic_and_transport"
        or record.get("development_run") != "85604"
        or record.get("held_out_85606_read") is not False
    ):
        raise ValueError("matched reconstruction result identity differs")
    identity = record.get("identity", {})
    family = str(identity.get("family", ""))
    codec = str(identity.get("codec", ""))
    seed = int(identity.get("seed", -1))
    if family not in {"c5p", "e6b"} or codec not in {"dcae_l20", "dcae_l10"}:
        raise ValueError("matched reconstruction family or codec differs")
    if seed not in {1701, 1702, 1703}:
        raise ValueError("matched reconstruction seed differs")
    if record.get("checkpoint_reload_bitwise_exact_on_evaluation_worker") is not True:
        raise ValueError("matched reconstruction reload gate did not pass")
    for split, interval in (
        ("training", TRAIN_INTERVAL),
        ("validation", VALIDATION_INTERVAL),
    ):
        item = record.get(split, {})
        if item.get("frames") != list(interval) or int(item.get("frame_count", -1)) != (
            interval[1] - interval[0]
        ):
            raise ValueError(f"matched reconstruction {split} interval differs")
        if not item.get("candidate_native81", {}).get("sha256"):
            raise ValueError(f"matched reconstruction {split} candidate is missing")
    return {
        "family": family,
        "codec": codec,
        "seed": seed,
        "checkpoint_sha256": str(identity["checkpoint_sha256"]),
        "evaluation_commit": str(record["evaluation_commit"]),
    }


def candidate_from_result(
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    split: str,
    interval: tuple[int, int],
) -> MatchedCandidateArtifact:
    item = record[split]["candidate_native81"]
    return MatchedCandidateArtifact(
        Path(item["path"]),
        sha256=str(item["sha256"]),
        family=str(identity["family"]),
        codec=str(identity["codec"]),
        seed=int(identity["seed"]),
        checkpoint_sha256=str(identity["checkpoint_sha256"]),
        frames=tuple(range(*interval)),
    )


def phi_from_result(
    path: Path,
    sha256: str,
    *,
    candidate: MatchedCandidateArtifact,
    interval: tuple[int, int],
) -> tuple[MatchedPhiArtifact, dict[str, Any]]:
    record = load_hash_locked_json(path, sha256)
    if (
        record.get("scope") != "phase2_matched_e6b_elliptic_output"
        or record.get("status") != "completed"
        or record.get("development_run") != "85604"
        or record.get("held_out_85606_read") is not False
        or record.get("truth_layout") is not False
        or record.get("frame_interval") != list(interval)
        or int(record.get("frame_count", -1)) != interval[1] - interval[0]
        or record.get("truth_replay_gate") is not None
        or record.get("source_input", {}).get("sha256") != candidate.sha256
    ):
        raise ValueError("matched E6B elliptic result identity differs")
    output = record.get("derived_phi", {})
    artifact = MatchedPhiArtifact(
        Path(output["path"]),
        sha256=str(output["sha256"]),
        source_candidate_sha256=candidate.sha256,
        frames=tuple(range(*interval)),
    )
    return artifact, record


def validate_truth_replay_summary(record: Mapping[str, Any]) -> None:
    if (
        record.get("scope") != "phase2_matched_e6b_zero_seed_truth_replay"
        or record.get("status") != "pass"
        or record.get("development_run") != "85604"
        or record.get("held_out_85606_read") is not False
        or record.get("coverage") != [0, 624]
        or int(record.get("frame_count", -1)) != 624
        or record.get("all_frames_passed") is not True
        or record.get("boundary_only_zero_interior_seed") is not True
        or record.get("zperiod") != 5
    ):
        raise ValueError("E6B zero-seed truth replay did not pass")


def main() -> int:
    args = parse_args()
    if not 1 <= args.chunk_frames <= 16:
        raise ValueError("chunk-frames must lie in 1..16")
    required_paths = (
        args.reconstruction_result,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.artifact_root,
        args.output,
    )
    for path in required_paths:
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    if args.output.exists():
        raise FileExistsError(args.output)

    started = datetime.now(timezone.utc)
    monotonic_started = time.monotonic()
    reconstruction = load_hash_locked_json(
        args.reconstruction_result,
        args.reconstruction_result_sha256,
    )
    identity = validate_reconstruction_result(reconstruction)
    train_candidate = candidate_from_result(
        reconstruction,
        identity,
        split="training",
        interval=TRAIN_INTERVAL,
    )
    validation_candidate = candidate_from_result(
        reconstruction,
        identity,
        split="validation",
        interval=VALIDATION_INTERVAL,
    )

    native_truth_record = load_hash_locked_json(
        args.native_truth_result,
        args.native_truth_result_sha256,
    )
    native_truth = NativeTruthCatalog(native_truth_record)
    geometry_manifest = load_hash_locked_json(
        args.geometry_manifest,
        args.geometry_manifest_sha256,
    )
    geometry = load_transport_geometry(
        geometry_path=args.geometry,
        geometry_manifest=geometry_manifest,
    )
    catalog = load_official_catalog(args.artifact_root)

    train_phi = None
    validation_phi = None
    phi_records = None
    truth_replay = None
    if identity["family"] == "e6b":
        required_e6b = (
            args.train_phi_result,
            args.train_phi_result_sha256,
            args.validation_phi_result,
            args.validation_phi_result_sha256,
            args.truth_replay_summary,
            args.truth_replay_summary_sha256,
        )
        if any(value is None for value in required_e6b):
            raise ValueError("E6B finalization requires phi outputs and truth replay")
        train_phi, train_phi_record = phi_from_result(
            args.train_phi_result,
            args.train_phi_result_sha256,
            candidate=train_candidate,
            interval=TRAIN_INTERVAL,
        )
        validation_phi, validation_phi_record = phi_from_result(
            args.validation_phi_result,
            args.validation_phi_result_sha256,
            candidate=validation_candidate,
            interval=VALIDATION_INTERVAL,
        )
        truth_replay = load_hash_locked_json(
            args.truth_replay_summary,
            args.truth_replay_summary_sha256,
        )
        validate_truth_replay_summary(truth_replay)
        phi_records = {
            "training": train_phi_record,
            "validation": validation_phi_record,
            "truth_replay": truth_replay,
        }
    elif any(
        value is not None
        for value in (
            args.train_phi_result,
            args.train_phi_result_sha256,
            args.validation_phi_result,
            args.validation_phi_result_sha256,
            args.truth_replay_summary,
            args.truth_replay_summary_sha256,
        )
    ):
        raise ValueError("C5P finalization refuses E6B elliptic inputs")

    if identity["family"] == "e6b":
        common_training = evaluate_e6b_common_interval(
            catalog=catalog,
            candidate=train_candidate,
            phi=train_phi,
            split="train",
            frames=tuple(range(*TRAIN_INTERVAL)),
            seed=int(identity["seed"]),
            chunk_frames=args.chunk_frames,
        )
        common_validation = evaluate_e6b_common_interval(
            catalog=catalog,
            candidate=validation_candidate,
            phi=validation_phi,
            split="validation",
            frames=tuple(range(*VALIDATION_INTERVAL)),
            seed=int(identity["seed"]),
            chunk_frames=args.chunk_frames,
        )
        common_materiality = training_materiality(common_training["overall"])
        common_gate = build_matched_o1_view_gate(
            validation_overall=common_validation["overall"],
            validation_blocks=common_validation["blocks"],
            materiality=common_materiality,
        )
        boundary_gate = {
            "training": common_training["boundary_bypass"],
            "validation": common_validation["boundary_bypass"],
            "passes": bool(
                common_training["boundary_bypass"]["passes"]
                and common_validation["boundary_bypass"]["passes"]
            ),
        }
        common_view = {
            "training": common_training,
            "training_materiality": common_materiality,
            "validation": common_validation,
            "gate": common_gate,
        }
    else:
        common_gate = reconstruction["native_view_pretransport_gate"]
        boundary_gate = {"applicable": False, "passes": True}
        common_view = {
            "identity": "same_as_c5p_native_view",
            "training": "see_reconstruction_result.training",
            "training_materiality": "see_reconstruction_result.training_materiality",
            "validation": "see_reconstruction_result.validation",
            "gate": common_gate,
        }

    transport_training = evaluate_matched_transport_interval(
        truth=native_truth,
        candidate=train_candidate,
        phi=train_phi,
        geometry=geometry,
        split="train",
        frames=tuple(range(*TRAIN_INTERVAL)),
        chunk_frames=args.chunk_frames,
    )
    transport_validation = evaluate_matched_transport_interval(
        truth=native_truth,
        candidate=validation_candidate,
        phi=validation_phi,
        geometry=geometry,
        split="validation",
        frames=tuple(range(*VALIDATION_INTERVAL)),
        chunk_frames=args.chunk_frames,
    )
    transport_gate = build_matched_o1_transport_gate(
        overall=transport_validation["overall"],
        temporal_blocks=transport_validation["blocks"],
    )

    native_gate = reconstruction["native_view_pretransport_gate"]
    conditions = {
        "native_predicted_view": bool(native_gate["passes"]),
        "common_transport_view": bool(common_gate["passes"]),
        "authoritative_native81_transport": bool(transport_gate["passes"]),
        "boundary_bypass": bool(boundary_gate["passes"]),
        "e6b_zero_seed_truth_replay": (
            bool(truth_replay["all_frames_passed"])
            if truth_replay is not None
            else True
        ),
        "shape_finiteness_and_provenance": True,
    }
    complete_pass = all(conditions.values())
    result = {
        "schema_version": 1,
        "scope": "phase2_matched_o1_codec_complete",
        "status": "pass" if complete_pass else "fail",
        "development_run": "85604",
        "held_out_85606_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "identity": identity,
        "source_reconstruction_result": reconstruction,
        "common_view": common_view,
        "boundary_gate": boundary_gate,
        "transport": {
            "training": transport_training,
            "validation": transport_validation,
            "gate": transport_gate,
        },
        "elliptic": phi_records,
        "complete_o1_gate": {
            "conditions": conditions,
            "passes": complete_pass,
            "status": "pass" if complete_pass else "fail",
            "aggregate_scores_cannot_override_a_failed_component": True,
        },
        "execution": {
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - monotonic_started,
            "chunk_frames": args.chunk_frames,
            "numpy": np.__version__,
            "peak_process_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "provenance": {
            "reconstruction_result": {
                "path": str(args.reconstruction_result.resolve(strict=True)),
                "sha256": args.reconstruction_result_sha256,
            },
            "native_truth_result": {
                "path": str(args.native_truth_result.resolve(strict=True)),
                "sha256": args.native_truth_result_sha256,
            },
            "geometry_manifest": {
                "path": str(args.geometry_manifest.resolve(strict=True)),
                "sha256": args.geometry_manifest_sha256,
            },
            "geometry": {
                "path": str(args.geometry.resolve(strict=True)),
                "sha256": sha256_path(args.geometry.resolve(strict=True)),
            },
            "official_model_artifact_root": str(args.artifact_root),
            "finalizer_sha256": sha256_path(Path(__file__).resolve()),
        },
    }
    output = args.output.resolve(strict=False)
    write_strict_json_atomic(output, _json_safe(result))
    print(f"wrote {output}", flush=True)
    # A failed scientific gate is a completed, usable result rather than an
    # infrastructure failure. The recorded status drives the R1 -> R2 ladder.
    return 0


if __name__ == "__main__":
    sys.exit(main())
