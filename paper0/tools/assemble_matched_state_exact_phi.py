#!/usr/bin/env python3
"""Verify and assemble exact-phi outputs for predicted E6B bounded forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from paper0.tools.generate_matched_state_bounded_forecasts import SCOPE as GENERATION_SCOPE
from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


SCOPE = "post_ecrd_old_85604_matched_state_exact_phi"
EXPECTED_CANDIDATES = (
    "h4_direct_predicted_e6b_native81.h5",
    "h4_autoregressive_lead1_predicted_e6b_native81.h5",
    "h4_autoregressive_lead2_predicted_e6b_native81.h5",
    "h8_direct_predicted_e6b_native81.h5",
    "h8_autoregressive_lead1_predicted_e6b_native81.h5",
    "h8_autoregressive_lead2_predicted_e6b_native81.h5",
    "h8_autoregressive_lead4_predicted_e6b_native81.h5",
)


def locked_json(path: Path, digest: str, *, label: str) -> dict[str, Any]:
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return load_strict_json(path)


def candidate_frame_interval(path: Path) -> list[int]:
    with h5py.File(path, "r") as handle:
        frames = np.asarray(handle["coordinates/frame_index"][:], dtype=np.int64)
        if (
            frames.ndim != 1
            or frames.size == 0
            or np.any(np.diff(frames) != 1)
        ):
            raise ValueError("elliptic candidate frames are not contiguous")
        if bool(handle.attrs.get("target_truth_used_during_generation", True)):
            raise ValueError("elliptic candidate reports target-truth use")
        if handle.attrs.get("boundary_policy") != (
            "predicted_Bphi_no_truth_bypass"
        ):
            raise ValueError("elliptic candidate boundary policy differs")
        if "phi" in handle["candidate"]:
            raise ValueError("elliptic candidate must not contain target phi")
    return [int(frames[0]), int(frames[-1]) + 1]


def assemble(
    *,
    generation_result: Path,
    generation_result_sha256: str,
    phi_root: Path,
    paper0_commit: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    generation = locked_json(
        generation_result,
        generation_result_sha256,
        label="E6B generation result",
    )
    if (
        generation.get("scope") != GENERATION_SCOPE
        or generation.get("status") != "completed"
        or generation.get("development_run") != "85604"
        or generation.get("family") != "e6b"
        or generation.get("held_out_85606_read") is not False
        or generation.get("new_nersc_data_read") is not False
        or generation.get("target_truth_used_during_generation") is not False
    ):
        raise ValueError("E6B generation result scope differs")
    candidates = generation.get("elliptic_candidates", [])
    names = tuple(Path(str(record.get("path", ""))).name for record in candidates)
    if names != EXPECTED_CANDIDATES:
        raise ValueError("E6B elliptic candidate set or order differs")

    outputs = []
    for candidate_record in candidates:
        candidate = Path(str(candidate_record["path"]))
        candidate_sha = str(candidate_record["sha256"])
        assert_development_path(candidate)
        if sha256_path(candidate) != candidate_sha:
            raise ValueError("E6B elliptic candidate SHA-256 differs")
        interval = candidate_frame_interval(candidate)
        task = phi_root / candidate.stem
        result_path = task / "result.json"
        result_sha = sha256_path(result_path)
        result = load_strict_json(result_path)
        if (
            result.get("scope") != "phase2_matched_e6b_elliptic_output"
            or result.get("status") != "completed"
            or result.get("development_run") != "85604"
            or result.get("held_out_85606_read") is not False
            or result.get("truth_layout") is not False
            or result.get("truth_replay_gate") is not None
            or result.get("frame_interval") != interval
            or result.get("source_input", {}).get("path") != str(candidate.resolve())
            or result.get("source_input", {}).get("sha256") != candidate_sha
        ):
            raise ValueError(f"exact-phi result differs for {candidate.name}")
        phi_path = Path(str(result.get("derived_phi", {}).get("path", "")))
        phi_sha = str(result.get("derived_phi", {}).get("sha256", ""))
        assert_development_path(phi_path)
        if not phi_sha or sha256_path(phi_path) != phi_sha:
            raise ValueError(f"exact-phi artifact differs for {candidate.name}")
        outputs.append(
            {
                "candidate": {
                    "path": str(candidate),
                    "sha256": candidate_sha,
                    "frame_interval": interval,
                },
                "elliptic_result": {
                    "path": str(result_path),
                    "sha256": result_sha,
                },
                "derived_phi": {"path": str(phi_path), "sha256": phi_sha},
            }
        )

    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": "completed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "target_truth_phi_read": False,
        "truth_layout": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "physics_scoring_performed": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
        "paper0_commit": paper0_commit,
        "slurm_job_id": slurm_job_id,
        "generation_result": {
            "path": str(generation_result),
            "sha256": generation_result_sha256,
        },
        "candidate_count": len(outputs),
        "outputs": outputs,
        "paired_common_view_physics_scoring_authorized": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-result", type=Path, required=True)
    parser.add_argument("--generation-result-sha256", required=True)
    parser.add_argument("--phi-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.generation_result, args.phi_root, args.output):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = assemble(
        generation_result=args.generation_result,
        generation_result_sha256=args.generation_result_sha256,
        phi_root=args.phi_root,
        paper0_commit=args.paper0_commit,
        slurm_job_id=args.slurm_job_id,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
