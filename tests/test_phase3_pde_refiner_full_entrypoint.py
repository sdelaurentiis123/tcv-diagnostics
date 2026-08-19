"""Static authorization tests for the full-only B4 training entrypoint."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.model_data import load_strict_json
from tcv_diagnostics.pde_refiner_training import RefinerParentArtifacts


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_b4_pde_refiner_full.py"
SMOKE_ENTRYPOINT = ROOT / "paper0/tools/train_b4_pde_refiner.py"
MANIFEST = ROOT / "paper0/manifests/phase3_b4_full_evaluation_85604.json"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "train_b4_pde_refiner_full",
        ENTRYPOINT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_artifacts(manifest: dict) -> RefinerParentArtifacts:
    parent = manifest["deterministic_parent"]
    codec = manifest["codec"]
    return RefinerParentArtifacts(
        checkpoint_path=Path(parent["checkpoint_path"]),
        checkpoint_sha256=parent["checkpoint_sha256"],
        codec_path=Path(codec["checkpoint_path"]),
        codec_sha256=codec["checkpoint_sha256"],
        latent_normalization_path=Path(codec["latent_normalization_path"]),
        latent_normalization_sha256=codec["latent_normalization_sha256"],
    )


def test_full_entrypoint_authorizes_exact_seed1701_contract() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    record = module.authorize_full_from_manifest(
        manifest,
        mode="full",
        seed=1701,
        artifacts=frozen_artifacts(manifest),
        manifest_path=MANIFEST,
    )
    assert record["authorized"] is True
    assert record["scope"] == "B4_PDE_Refiner_H1_seed1701_full_training_85604"
    assert record["development_run"] == "85604"
    assert record["held_out_85606_read"] is False
    assert record["full_B4_training_authorized"] is True
    assert record["scientific_result"] is False
    assert record["training_complete_is_scientific_acceptance"] is False
    assert record["H_det_evaluated"] is False
    assert record["H_prob_evaluated"] is False
    assert record["seed"] == 1701
    assert record["passing_smoke"]["job_id"] == "6899469"


def test_full_entrypoint_rejects_mode_seed_scope_and_budget_drift() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    artifacts = frozen_artifacts(manifest)
    with pytest.raises(RuntimeError, match="only mode='full'"):
        module.authorize_full_from_manifest(
            manifest,
            mode="smoke",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )
    with pytest.raises(RuntimeError, match="only seed 1701"):
        module.authorize_full_from_manifest(
            manifest,
            mode="full",
            seed=1702,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )

    held_out = deepcopy(manifest)
    held_out["held_out_85606_access_allowed"] = True
    with pytest.raises(RuntimeError, match="held-out"):
        module.authorize_full_from_manifest(
            held_out,
            mode="full",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )
    changed_budget = deepcopy(manifest)
    changed_budget["training"]["epochs"] = 101
    with pytest.raises(RuntimeError, match="epochs"):
        module.authorize_full_from_manifest(
            changed_budget,
            mode="full",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )


def test_full_entrypoint_rejects_artifact_selection_and_smoke_drift() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    artifacts = frozen_artifacts(manifest)
    bad_parent = RefinerParentArtifacts(
        checkpoint_path=artifacts.checkpoint_path,
        checkpoint_sha256="0" * 64,
        codec_path=artifacts.codec_path,
        codec_sha256=artifacts.codec_sha256,
        latent_normalization_path=artifacts.latent_normalization_path,
        latent_normalization_sha256=artifacts.latent_normalization_sha256,
    )
    with pytest.raises(RuntimeError, match="parent hash"):
        module.authorize_full_from_manifest(
            manifest,
            mode="full",
            seed=1701,
            artifacts=bad_parent,
            manifest_path=MANIFEST,
        )

    changed_selection = deepcopy(manifest)
    changed_selection["checkpoint_selection"]["ensemble_members"] = 3
    with pytest.raises(RuntimeError, match="checkpoint-selection"):
        module.authorize_full_from_manifest(
            changed_selection,
            mode="full",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )

    forged_smoke = deepcopy(manifest)
    forged_smoke["evidence_locks"]["B4_smoke"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="evidence hash"):
        module.authorize_full_from_manifest(
            forged_smoke,
            mode="full",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )


def test_full_and_smoke_entrypoints_remain_disjoint() -> None:
    full_source = ENTRYPOINT.read_text(encoding="utf-8")
    smoke_source = SMOKE_ENTRYPOINT.read_text(encoding="utf-8")
    assert 'choices=("full",)' in full_source
    assert "train_pde_refiner_full" in full_source
    assert "train_pde_refiner_smoke" not in full_source
    assert 'choices=("smoke",)' in smoke_source
    assert "train_pde_refiner_smoke" in smoke_source
    assert "train_pde_refiner_full" not in smoke_source
    for source in (full_source, smoke_source):
        assert 'torch.set_float32_matmul_precision("highest")' in source
        assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
