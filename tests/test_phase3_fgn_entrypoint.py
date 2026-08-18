"""Static authorization tests for the bounded B3 FGN smoke entrypoint."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.fgn_training import ParentArtifacts
from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_b3_fgn.py"
MANIFEST = ROOT / "paper0/manifests/phase3_b3_fgn_85604.json"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("train_b3_fgn", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_artifacts(manifest: dict) -> ParentArtifacts:
    parent = manifest["deterministic_parent"]
    codec = manifest["codec"]
    return ParentArtifacts(
        checkpoint_path=Path(parent["checkpoint_path"]),
        checkpoint_sha256=parent["checkpoint_sha256"],
        codec_path=Path(codec["checkpoint_path"]),
        codec_sha256=codec["checkpoint_sha256"],
        latent_normalization_path=Path(codec["latent_normalization_path"]),
        latent_normalization_sha256=codec["latent_normalization_sha256"],
    )


def test_entrypoint_authorizes_only_exact_seed_1701_smoke() -> None:
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
    assert record["full_B3_training_authorized"] is False

    with pytest.raises(RuntimeError, match="only the bounded"):
        module.authorize_from_manifest(
            manifest,
            mode="full",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )


def test_entrypoint_rejects_scope_time_and_parent_drift() -> None:
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

    time_leak = deepcopy(manifest)
    time_leak["data"]["absolute_time_input_allowed"] = True
    with pytest.raises(RuntimeError, match="prohibited data flag"):
        module.authorize_from_manifest(
            time_leak,
            mode="smoke",
            seed=1701,
            artifacts=artifacts,
            manifest_path=MANIFEST,
        )

    bad_parent = ParentArtifacts(
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


def test_entrypoint_cli_does_not_offer_full_mode() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'choices=("smoke",)' in source
    assert "train_fgn_smoke" in source
    assert "train_fgn_full" not in source
