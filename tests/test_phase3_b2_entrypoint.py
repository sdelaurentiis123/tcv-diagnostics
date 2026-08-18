"""Static authorization tests for the B2 smoke entrypoint."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_b2_ldm.py"
MANIFEST = ROOT / "paper0/manifests/phase3_b2_ldm_85604.json"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("train_b2_ldm", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_authorizes_only_the_exact_seed_1701_smoke() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    checkpoint = manifest["codec"]["selected_checkpoints"][0]
    record = module.authorize_from_manifest(
        manifest,
        mode="smoke",
        seed=1701,
        codec_checkpoint=Path(checkpoint["path"]),
        codec_sha256=checkpoint["sha256"],
        manifest_path=MANIFEST,
    )
    assert record["authorized"] is True
    assert record["held_out_85606_read"] is False
    assert record["full_B2_training_authorized"] is False

    with pytest.raises(RuntimeError, match="only the bounded"):
        module.authorize_from_manifest(
            manifest,
            mode="full",
            seed=1701,
            codec_checkpoint=Path(checkpoint["path"]),
            codec_sha256=checkpoint["sha256"],
            manifest_path=MANIFEST,
        )


def test_entrypoint_rejects_manifest_scope_expansion_and_codec_drift() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    checkpoint = manifest["codec"]["selected_checkpoints"][0]

    expanded = deepcopy(manifest)
    expanded["full_training_authorized"] = True
    with pytest.raises(RuntimeError, match="full training"):
        module.authorize_from_manifest(
            expanded,
            mode="smoke",
            seed=1701,
            codec_checkpoint=Path(checkpoint["path"]),
            codec_sha256=checkpoint["sha256"],
            manifest_path=MANIFEST,
        )

    with pytest.raises(RuntimeError, match="codec hash"):
        module.authorize_from_manifest(
            manifest,
            mode="smoke",
            seed=1701,
            codec_checkpoint=Path(checkpoint["path"]),
            codec_sha256="0" * 64,
            manifest_path=MANIFEST,
        )


def test_entrypoint_cli_does_not_offer_full_mode() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'choices=("smoke",)' in source
    assert "train_b2_smoke" in source

