"""Protocol and launcher tests for the bounded axial GPU smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper0.tools.smoke_axial_operator import authorize_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/post_ecrd_old_85604_axial_operator_smoke.json"
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_axial_operator_smoke.sbatch"


def test_axial_smoke_manifest_is_bounded_and_authorized() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    authorize_manifest(manifest)
    assert manifest["state"]["auxiliary_context_fields"] == ["phi"]
    assert manifest["architecture"]["official_GAOT_reproduction"] is False
    assert manifest["optimization"]["optimizer_steps"] == 2
    assert manifest["loss"]["physics_derived_quantities_used"] is False


def test_axial_smoke_manifest_fails_closed_on_scientific_training() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["scientific_training_authorized"] = True
    with pytest.raises(ValueError, match="scientific training"):
        authorize_manifest(manifest)


def test_axial_smoke_launcher_requires_clean_commit_and_wandb() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_EXPECTED_COMMIT" in text
    assert "status --porcelain --untracked-files=all" in text
    assert "WANDB_MODE=online" in text
    assert "--gres=gpu:1" in text
    assert "smoke_axial_operator.py" in text
