"""Static fail-closed tests for the full B5 training entrypoint."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_b5_field_residual_edm_full.py"
MANIFEST = ROOT / "paper0/manifests/phase3_b5_full_training_evaluation_85604.json"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "train_b5_field_residual_edm_full", ENTRYPOINT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authorize(module, manifest: dict) -> dict:
    locks = manifest["evidence_locks"]
    return module.authorize_full_from_manifest(
        manifest,
        manifest_path=MANIFEST,
        mode="full",
        seed=1701,
        train_forecast_sha256=locks["H1_training_forecast"]["sha256"],
        validation_forecast_sha256=locks["H1_validation_forecast"]["sha256"],
        residual_audit_sha256=locks["residual_audit"]["sha256"],
    )


def test_B5_full_entrypoint_authorizes_only_exact_seed1701_training() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)
    record = authorize(module, manifest)
    assert record["authorized"] is True
    assert record["scope"] == (
        "B5_seed1701_full_training_and_data_only_selection_85604"
    )
    assert record["development_run"] == "85604"
    assert record["blind_test_read"] is False
    assert record["scientific_result"] is False
    assert record["scientific_forecast_generated"] is False
    with pytest.raises(RuntimeError, match="only the frozen"):
        module.authorize_full_from_manifest(
            manifest,
            manifest_path=MANIFEST,
            mode="smoke",
            seed=1701,
            train_forecast_sha256=manifest["evidence_locks"]["H1_training_forecast"][
                "sha256"
            ],
            validation_forecast_sha256=manifest["evidence_locks"][
                "H1_validation_forecast"
            ]["sha256"],
            residual_audit_sha256=manifest["evidence_locks"]["residual_audit"][
                "sha256"
            ],
        )


def test_B5_full_entrypoint_rejects_scope_data_schedule_and_evidence_drift() -> None:
    module = load_entrypoint()
    manifest = load_strict_json(MANIFEST)

    opened = deepcopy(manifest)
    opened["held_out_85606_access_allowed"] = True
    with pytest.raises(RuntimeError, match="sequestered scope"):
        authorize(module, opened)

    time_leak = deepcopy(manifest)
    time_leak["data"]["absolute_time_input_allowed"] = True
    with pytest.raises(RuntimeError, match="data field"):
        authorize(module, time_leak)

    changed_schedule = deepcopy(manifest)
    changed_schedule["full_training"]["peak_learning_rate"] = 2.0e-4
    with pytest.raises(RuntimeError, match="training field"):
        authorize(module, changed_schedule)

    physics_selection = deepcopy(manifest)
    physics_selection["checkpoint_selection"]["physics_metric_allowed"] = True
    with pytest.raises(RuntimeError, match="checkpoint-selection"):
        authorize(module, physics_selection)

    wrong_forecast = deepcopy(manifest)
    wrong_forecast["evidence_locks"]["H1_validation_forecast"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="validation-forecast lock"):
        authorize(module, wrong_forecast)


def test_B5_full_entrypoint_uses_two_strict_mean_readers_and_no_sampler() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'choices=("full",)' in source
    assert "train_b5_edm_full" in source
    assert "B5TrainingForecastArtifact" in source
    assert "O2ForecastArtifact" in source
    assert 'split="train"' in source
    assert 'split="validation"' in source
    assert "augment=False" in source
    assert "B5EDMFullOnlineWandbTracker.start" in source
    assert 'if "H100" not in accelerator' in source
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert ".sample_normalized(" not in source
    assert "scientific_sampler_seed_bank(" not in source
