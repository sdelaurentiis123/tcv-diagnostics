"""Manifest-authority tests for the matched ECRD training entrypoint."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tcv_diagnostics.ecrd_training import frozen_parameter_counts
from tcv_diagnostics.models.ecrd import MultiscaleNoiseConfig


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_ecrd.py"
SPEC = importlib.util.spec_from_file_location("train_ecrd_entrypoint", ENTRYPOINT)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)
LOCKS = {
    "H1_training_parent": "a" * 64,
    "H1_validation_parent": "b" * 64,
    "sym_H1_training_parent": "c" * 64,
    "sym_H1_validation_parent": "d" * 64,
}


def _manifest(*, mode: str) -> dict[str, object]:
    smoke = mode == "smoke"
    return {
        "status": (
            "frozen_before_ECRD_engineering_smoke"
            if smoke
            else "frozen_after_passing_ECRD_smoke_before_full_training"
        ),
        "development_run": "85604",
        "held_out_85606_access_allowed": False,
        "protocol": {"sha256": TRAIN.EXPECTED_PROTOCOL_SHA256},
        "authorized_arms": ["B5", "B5-Context", "ECRD", "ECRD-History"],
        "authorized_seeds": [1701] if smoke else [1701, 1702, 1703],
        "full_training_authorized": not smoke,
        "symmetrized_parent_use": (
            {
                "artifact_authority": "bounded_non_scientific_engineering_smoke_only",
                "execution_device": "cpu-smoke",
                "authorized_modes": ["smoke"],
                "H100_comparison_required_before_full_training": True,
            }
            if smoke
            else {
                "artifact_authority": "scientific_H100_parent",
                "execution_device": "h100",
                "authorized_modes": ["smoke", "full"],
            }
        ),
        "evidence_locks": {
            **{name: {"sha256": value} for name, value in LOCKS.items()},
            "model_dataset": {
                "manifest_sha256": "e" * 64,
                "normalization_sha256": "f" * 64,
                "artifact_index_sha256": "1" * 64,
            },
        },
        "data": {
            "training_targets": [2, 432],
            "guard_frames": [432, 496],
            "validation_targets": [498, 624],
            "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
            "periodic_axes_xyz": [False, False, True],
            "zperiod": 5,
            "mode_mapping": "n=5k",
        },
        "smoke": {
            "training_targets": [2, 6],
            "validation_targets": [498, 502],
            "epochs": 1,
            "optimizer_steps": 2,
            "scientific_result": False,
        },
        "exact_implementation": {
            "parameter_counts": frozen_parameter_counts(),
            "multiscale_noise": MultiscaleNoiseConfig().to_record(),
        },
        "smoke_evidence": {"all_four_arms_passed": True},
    }


def _authorize(tmp_path: Path, manifest: dict[str, object], *, mode: str):
    path = tmp_path / f"{mode}.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return TRAIN.authorize_manifest(
        manifest,
        manifest_path=path,
        manifest_sha256=digest,
        mode=mode,
        arm="ECRD",
        seed=1701,
        input_hashes=LOCKS,
    )


def test_smoke_requires_bounded_cpu_parent_authority(tmp_path: Path) -> None:
    manifest = _manifest(mode="smoke")
    authorization = _authorize(tmp_path, manifest, mode="smoke")
    assert authorization["scope"] == "ECRD_smoke_ECRD_seed1701_85604"
    manifest["symmetrized_parent_use"] = {
        "artifact_authority": "scientific_H100_parent",
        "execution_device": "h100",
        "authorized_modes": ["smoke", "full"],
    }
    with pytest.raises(RuntimeError, match="smoke parent-use"):
        _authorize(tmp_path, manifest, mode="smoke")


def test_full_training_requires_scientific_h100_parent_authority(
    tmp_path: Path,
) -> None:
    manifest = _manifest(mode="full")
    authorization = _authorize(tmp_path, manifest, mode="full")
    assert authorization["scope"] == "ECRD_full_ECRD_seed1701_85604"
    manifest["symmetrized_parent_use"] = {
        "artifact_authority": "bounded_non_scientific_engineering_smoke_only",
        "execution_device": "cpu-smoke",
        "authorized_modes": ["smoke"],
        "H100_comparison_required_before_full_training": True,
    }
    with pytest.raises(RuntimeError, match="full-training parent authority"):
        _authorize(tmp_path, manifest, mode="full")
