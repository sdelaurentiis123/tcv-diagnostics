"""Static and known-answer authorization tests for full B2 training."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_b2_ldm_full.py"
FULL_MANIFEST = ROOT / "paper0/manifests/phase3_b2_full_evaluation_85604.json"
IMPLEMENTATION_MANIFEST = ROOT / "paper0/manifests/phase3_b2_ldm_85604.json"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("train_b2_ldm_full", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_for(seed: int) -> dict:
    implementation = load_strict_json(IMPLEMENTATION_MANIFEST)
    return next(
        item
        for item in implementation["codec"]["selected_checkpoints"]
        if int(item["seed"]) == int(seed)
    )


@pytest.mark.parametrize("seed", [1701, 1702, 1703])
def test_full_entrypoint_authorizes_exact_three_seed_matrix(seed: int) -> None:
    module = load_entrypoint()
    manifest = load_strict_json(FULL_MANIFEST)
    checkpoint = checkpoint_for(seed)
    record, implementation = module.authorize_full_from_manifest(
        manifest,
        mode="full",
        seed=seed,
        codec_checkpoint=Path(checkpoint["path"]),
        codec_sha256=checkpoint["sha256"],
        manifest_path=FULL_MANIFEST,
    )
    assert record["authorized"] is True
    assert record["development_run"] == "85604"
    assert record["held_out_85606_read"] is False
    assert record["full_B2_training_authorized"] is True
    assert record["probabilistic_scientific_gate_evaluated"] is False
    assert record["seed"] == seed
    assert implementation["data"]["zperiod"] == 5
    assert implementation["data"]["mode_mapping"] == "n=5k"


def test_full_entrypoint_rejects_scope_budget_and_codec_drift() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(FULL_MANIFEST)
    checkpoint = checkpoint_for(1701)

    expanded = deepcopy(manifest)
    expanded["held_out_85606_access_allowed"] = True
    with pytest.raises(RuntimeError, match="held-out"):
        module.authorize_full_from_manifest(
            expanded,
            mode="full",
            seed=1701,
            codec_checkpoint=Path(checkpoint["path"]),
            codec_sha256=checkpoint["sha256"],
            manifest_path=FULL_MANIFEST,
        )

    changed_budget = deepcopy(manifest)
    changed_budget["training"]["epochs"] = 201
    with pytest.raises(RuntimeError, match="epochs"):
        module.authorize_full_from_manifest(
            changed_budget,
            mode="full",
            seed=1701,
            codec_checkpoint=Path(checkpoint["path"]),
            codec_sha256=checkpoint["sha256"],
            manifest_path=FULL_MANIFEST,
        )

    with pytest.raises(RuntimeError, match="codec hash"):
        module.authorize_full_from_manifest(
            manifest,
            mode="full",
            seed=1701,
            codec_checkpoint=Path(checkpoint["path"]),
            codec_sha256="0" * 64,
            manifest_path=FULL_MANIFEST,
        )


def test_full_entrypoint_cli_is_separate_and_full_only() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'choices=("full",)' in source
    assert "train_b2_full" in source
    assert "train_b2_smoke" not in source
    assert "85606" in source
