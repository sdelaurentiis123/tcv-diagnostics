"""Static fail-closed tests for the bounded B5 residual-EDM entrypoint."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_b5_field_residual_edm_smoke.py"
MANIFEST = (
    ROOT / "paper0/manifests/phase3_b5_field_residual_edm_smoke_85604.json"
)


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "train_b5_field_residual_edm_smoke", ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authorize(module, manifest: dict) -> dict:
    locks = manifest["evidence_locks"]
    return module.authorize_from_manifest(
        manifest,
        manifest_path=MANIFEST,
        mode="smoke",
        seed=1701,
        forecast_sha256=locks["H1_training_forecast"]["sha256"],
        residual_audit_sha256=locks["residual_audit"]["sha256"],
    )


def test_B5_entrypoint_authorizes_only_the_exact_bounded_smoke() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    record = authorize(module, manifest)
    assert record["authorized"] is True
    assert record["development_run"] == "85604"
    assert record["blind_test_read"] is False
    assert record["validation_read"] is False
    assert record["scientific_result"] is False
    assert record["full_training_authorized"] is False
    with pytest.raises(RuntimeError, match="only the bounded"):
        module.authorize_from_manifest(
            manifest,
            manifest_path=MANIFEST,
            mode="full",
            seed=1701,
            forecast_sha256=manifest["evidence_locks"]["H1_training_forecast"][
                "sha256"
            ],
            residual_audit_sha256=manifest["evidence_locks"]["residual_audit"][
                "sha256"
            ],
        )


def test_B5_entrypoint_rejects_access_input_model_and_evidence_drift() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)

    expanded = deepcopy(manifest)
    expanded["held_out_85606_access_allowed"] = True
    with pytest.raises(RuntimeError, match="held-out access"):
        authorize(module, expanded)

    time_leak = deepcopy(manifest)
    time_leak["data"]["absolute_time_input_allowed"] = True
    with pytest.raises(RuntimeError, match="prohibited data flag"):
        authorize(module, time_leak)

    compressed = deepcopy(manifest)
    compressed["model"]["DCAE_or_latent_representation_allowed"] = True
    with pytest.raises(RuntimeError, match="representation or boundary"):
        authorize(module, compressed)

    wrong_forecast = deepcopy(manifest)
    wrong_forecast["evidence_locks"]["H1_training_forecast"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="training-forecast hash"):
        authorize(module, wrong_forecast)


def test_B5_entrypoint_has_no_full_training_or_validation_path() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'choices=("smoke",)' in source
    assert "train_b5_edm_smoke" in source
    assert "OneStepWindowDataset" in source
    assert 'split="train"' in source
    assert "B5TrainingForecastArtifact" in source
    assert "B5EDMOnlineWandbTracker.start" in source
    assert 'if "H100" not in accelerator' in source
    assert 'torch.backends.cuda.matmul.allow_tf32 = False' in source
    assert "train_b5_edm_full" not in source
    assert 'split="validation"' not in source
