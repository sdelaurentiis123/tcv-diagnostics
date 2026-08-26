#!/usr/bin/env python3
"""Freeze the paired old-85604 state-view physics evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from paper0.tools.assemble_matched_state_exact_phi import SCOPE as PHI_SCOPE
from paper0.tools.generate_matched_state_bounded_forecasts import (
    FAMILIES,
    HORIZONS,
    SCOPE as GENERATION_SCOPE,
)
from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from tcv_diagnostics.bounded_rollout import method_schedule
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


SCOPE = "post_ecrd_old_85604_matched_state_physics_scoring"
PROTOCOL = (
    "paper0/protocol/"
    "POST_ECRD_OLD_85604_MATCHED_STATE_PHYSICS_FREEZE_2026-08-26.md"
)
ARTIFACT_ROOT = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/"
    "phase2_model_dataset/job_6893525"
)
CANONICAL_DEPENDENCIES = {
    "native_truth_result": {
        "path": "paper0/results/phase2_potential_vorticity_all_frame_6893033.json",
        "sha256": "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae",
    },
    "geometry_manifest": {
        "path": "paper0/manifests/phase2_85604_geometry_units.json",
        "sha256": "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460",
    },
    "geometry": {
        "path": "/mnt/ceph/users/sdelaurentiis/tcv-fresh-proj/85604/tcv_85604_adjusted.nc",
        "sha256": "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
    },
    "model_dataset_manifest": {
        "path": str(ARTIFACT_ROOT / "model_dataset_manifest.json"),
        "sha256": "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18",
    },
    "normalization": {
        "path": str(ARTIFACT_ROOT / "normalization.json"),
        "sha256": "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7",
    },
    "model_dataset_artifact_index": {
        "path": str(ARTIFACT_ROOT / "artifact_sha256.txt"),
        "sha256": "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01",
    },
}


def locked_json(path: Path, digest: str, *, label: str) -> dict[str, Any]:
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return load_strict_json(path)


def verify_generation(
    path: Path,
    digest: str,
    *,
    family: str,
    generation_manifest: Path,
    generation_manifest_sha256: str,
) -> dict[str, Any]:
    result = locked_json(path, digest, label=f"{family} generation result")
    if (
        result.get("scope") != GENERATION_SCOPE
        or result.get("status") != "completed"
        or result.get("development_run") != "85604"
        or result.get("family") != family
        or result.get("held_out_85606_read") is not False
        or result.get("new_nersc_data_read") is not False
        or result.get("guard_frames_read") is not False
        or result.get("training_performed") is not False
        or result.get("checkpoint_selection_performed") is not False
        or result.get("physics_evaluation_performed") is not False
        or result.get("target_truth_used_during_generation") is not False
        or result.get("manifest", {}).get("path") != str(generation_manifest)
        or result.get("manifest", {}).get("sha256")
        != generation_manifest_sha256
    ):
        raise ValueError(f"{family} generation-result contract differs")
    forecast = result.get("forecast", {})
    forecast_path = Path(str(forecast.get("path", "")))
    forecast_sha = str(forecast.get("sha256", ""))
    assert_development_path(forecast_path)
    if not forecast_sha or sha256_path(forecast_path) != forecast_sha:
        raise ValueError(f"{family} bounded forecast SHA-256 differs")
    if family == "c5p" and result.get("elliptic_candidates") != []:
        raise ValueError("C5P generation unexpectedly has elliptic candidates")
    if family == "e6b" and len(result.get("elliptic_candidates", [])) != 7:
        raise ValueError("E6B generation does not have seven candidates")
    return result


def freeze_manifest(
    *,
    generation_manifest: Path,
    generation_manifest_sha256: str,
    generation_results: Mapping[str, tuple[Path, str]],
    exact_phi_result: Path,
    exact_phi_result_sha256: str,
    paper0_root: Path,
    paper0_commit: str,
    dependencies: Mapping[str, Mapping[str, str]] = CANONICAL_DEPENDENCIES,
) -> dict[str, Any]:
    manifest = locked_json(
        generation_manifest,
        generation_manifest_sha256,
        label="bounded-generation manifest",
    )
    if (
        manifest.get("scope") != GENERATION_SCOPE
        or manifest.get("status")
        != "frozen_after_paired_transition_reduction_before_inference"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("new_nersc_data_access_allowed") is not False
        or manifest.get("guard_frames_read_allowed") is not False
        or manifest.get("training_allowed") is not False
        or manifest.get("checkpoint_selection_allowed") is not False
    ):
        raise ValueError("bounded-generation manifest contract differs")
    if set(generation_results) != set(FAMILIES):
        raise ValueError("generation-result family set differs")
    generated = {
        family: verify_generation(
            *generation_results[family],
            family=family,
            generation_manifest=generation_manifest,
            generation_manifest_sha256=generation_manifest_sha256,
        )
        for family in FAMILIES
    }

    phi = locked_json(
        exact_phi_result,
        exact_phi_result_sha256,
        label="exact-phi result",
    )
    e6b_path, e6b_sha = generation_results["e6b"]
    if (
        phi.get("scope") != PHI_SCOPE
        or phi.get("status") != "completed"
        or phi.get("development_run") != "85604"
        or phi.get("held_out_85606_read") is not False
        or phi.get("new_nersc_data_read") is not False
        or phi.get("target_truth_phi_read") is not False
        or phi.get("truth_layout") is not False
        or int(phi.get("candidate_count", -1)) != 7
        or phi.get("paired_common_view_physics_scoring_authorized") is not True
        or phi.get("generation_result", {}).get("path") != str(e6b_path)
        or phi.get("generation_result", {}).get("sha256") != e6b_sha
    ):
        raise ValueError("exact-phi result does not authorize paired scoring")

    dependency_locks: dict[str, dict[str, str]] = {}
    for label, record in dependencies.items():
        raw_path = Path(str(record["path"]))
        path = raw_path if raw_path.is_absolute() else paper0_root / raw_path
        assert_development_path(path)
        digest = str(record["sha256"])
        if sha256_path(path) != digest:
            raise ValueError(f"{label} SHA-256 differs")
        dependency_locks[label] = {"path": str(path), "sha256": digest}

    blocks = {
        "4": [[500, 541], [541, 582], [582, 624]],
        "8": [[504, 544], [544, 584], [584, 624]],
    }
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": "frozen_after_causal_exact_phi_before_paired_physics",
        "protocol": PROTOCOL,
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
        "paper0_commit_at_freeze": paper0_commit,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "state_views": ["c5p", "e6b"],
        "common_fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "generation_manifest": {
            "path": str(generation_manifest),
            "sha256": generation_manifest_sha256,
        },
        "generation_results": {
            family: {
                "path": str(generation_results[family][0]),
                "sha256": generation_results[family][1],
                "forecast": dict(generated[family]["forecast"]),
            }
            for family in FAMILIES
        },
        "exact_phi_result": {
            "path": str(exact_phi_result),
            "sha256": exact_phi_result_sha256,
        },
        "dependencies": dependency_locks,
        "evaluation": {
            "horizons": list(HORIZONS),
            "methods": {
                str(horizon): method_schedule(horizon) for horizon in HORIZONS
            },
            "target_frame_blocks": blocks,
            "common_normalization": "training_only_C5P_scalar_normalization",
            "e6b_phi": "pinned_exact_elliptic_from_predictions_only",
            "e6b_vi": "NVi/(2*softFloor(Ne,1e-7))",
            "physics_diagnostics_are_evaluation_only": True,
        },
        "decision": {
            "separatrix_transport_e6b_over_c5p_max": 0.90,
            "complex_cross_spectrum_e6b_strictly_better": True,
            "shared_state_e6b_over_c5p_max": 1.10,
            "spectral_power_e6b_over_c5p_max": 1.10,
            "all_primary_scalars_finite_required": True,
            "causal_exact_phi_required": True,
        },
        "wandb_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--generation-manifest-sha256", required=True)
    parser.add_argument("--c5p-generation-result", type=Path, required=True)
    parser.add_argument("--c5p-generation-result-sha256", required=True)
    parser.add_argument("--e6b-generation-result", type=Path, required=True)
    parser.add_argument("--e6b-generation-result-sha256", required=True)
    parser.add_argument("--exact-phi-result", type=Path, required=True)
    parser.add_argument("--exact-phi-result-sha256", required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = freeze_manifest(
        generation_manifest=args.generation_manifest,
        generation_manifest_sha256=args.generation_manifest_sha256,
        generation_results={
            "c5p": (
                args.c5p_generation_result,
                args.c5p_generation_result_sha256,
            ),
            "e6b": (
                args.e6b_generation_result,
                args.e6b_generation_result_sha256,
            ),
        },
        exact_phi_result=args.exact_phi_result,
        exact_phi_result_sha256=args.exact_phi_result_sha256,
        paper0_root=args.paper0_root,
        paper0_commit=args.paper0_commit,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
