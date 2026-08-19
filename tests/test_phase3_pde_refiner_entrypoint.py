"""Static authorization tests for the bounded B4 smoke entrypoint."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.model_data import load_strict_json
from tcv_diagnostics.pde_refiner_training import RefinerParentArtifacts


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_b4_pde_refiner.py"
MANIFEST = ROOT / "paper0/manifests/phase3_b4_pde_refiner_85604.json"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("train_b4_pde_refiner", ENTRYPOINT)
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


def test_entrypoint_authorizes_only_exact_seed1701_smoke() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    artifacts = frozen_artifacts(manifest)
    record = module.authorize_from_manifest(
        manifest,
        mode="smoke",
        seed=1701,
        artifacts=artifacts,
        manifest_path=MANIFEST,
    )
    assert record["authorized"] is True
    assert record["held_out_85606_read"] is False
    assert record["scientific_result"] is False
    assert record["full_B4_training_authorized"] is False
    with pytest.raises(RuntimeError, match="only the bounded"):
        module.authorize_from_manifest(
            manifest,
            mode="full",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )


def test_entrypoint_rejects_scope_data_precision_and_parent_drift() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    artifacts = frozen_artifacts(manifest)

    expanded = deepcopy(manifest)
    expanded["full_training_authorized"] = True
    with pytest.raises(RuntimeError, match="full training"):
        module.authorize_from_manifest(
            expanded,
            mode="smoke",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )

    future_leak = deepcopy(manifest)
    future_leak["data"]["future_truth_input_allowed_during_forecast"] = True
    with pytest.raises(RuntimeError, match="prohibited data flag"):
        module.authorize_from_manifest(
            future_leak,
            mode="smoke",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )

    low_precision = deepcopy(manifest)
    low_precision["precision"]["training"] = "bfloat16_autocast"
    with pytest.raises(RuntimeError, match="precision contract"):
        module.authorize_from_manifest(
            low_precision,
            mode="smoke",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )

    bad_parent = RefinerParentArtifacts(
        checkpoint_path=artifacts.checkpoint_path,
        checkpoint_sha256="0" * 64,
        codec_path=artifacts.codec_path,
        codec_sha256=artifacts.codec_sha256,
        latent_normalization_path=artifacts.latent_normalization_path,
        latent_normalization_sha256=artifacts.latent_normalization_sha256,
    )
    with pytest.raises(RuntimeError, match="parent hash"):
        module.authorize_from_manifest(
            manifest,
            mode="smoke",
            seed=1701,
            artifacts=bad_parent,
            manifest_path=MANIFEST,
        )


def test_entrypoint_cli_exposes_no_full_training_path() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'choices=("smoke",)' in source
    assert "train_pde_refiner_smoke" in source
    assert "train_pde_refiner_full" not in source
    assert 'torch.set_float32_matmul_precision("highest")' in source
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert "PDERefinerOnlineWandbTracker.start" in source
