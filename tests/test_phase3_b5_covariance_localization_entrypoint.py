"""Network-free checks for the read-only B5 localization entrypoint."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pytest

from paper0.tools.localize_b5_covariance import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    _block_index,
    _block_name,
    _legacy_training_batch_record,
    _legacy_training_cross_check,
    validate_authority,
)
from tcv_diagnostics.codec_training import sha256_path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/phase3_b5_covariance_localization_85604.json"
PROTOCOL = ROOT / "paper0/protocol/PHASE3_B5_COVARIANCE_LOCALIZATION_PROTOCOL.md"


def _authority_args(record: dict) -> Namespace:
    locks = record["evidence_locks"]

    def path(name: str) -> Path:
        value = Path(locks[name]["path"])
        return value if value.is_absolute() else ROOT / value

    return Namespace(
        localization_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        localization_protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        artifact_root=Path(locks["model_dataset"]["root"]),
        b5_forecast=path("B5_forecast"),
        b5_forecast_sha256=locks["B5_forecast"]["sha256"],
        b5_seed_bank=path("B5_scientific_sampler_seed_bank"),
        b5_seed_bank_sha256=locks["B5_scientific_sampler_seed_bank"]["sha256"],
        h1_forecast=path("H1_validation_forecast"),
        h1_forecast_sha256=locks["H1_validation_forecast"]["sha256"],
        b5_score=path("B5_score"),
        b5_score_sha256=locks["B5_score"]["sha256"],
        b5_gate=path("B5_final_gate"),
        b5_gate_sha256=locks["B5_final_gate"]["sha256"],
        training_audit=path("training_residual_audit"),
        training_audit_sha256=locks["training_residual_audit"]["sha256"],
        training_raw=path("training_residual_sufficient_statistics"),
        training_raw_sha256=locks["training_residual_sufficient_statistics"]["sha256"],
        h1_training_forecast=path("H1_training_forecast"),
        h1_training_forecast_sha256=locks["H1_training_forecast"]["sha256"],
        native_truth_result=path("native_truth_result"),
        native_truth_result_sha256=locks["native_truth_result"]["sha256"],
        geometry_manifest=path("geometry_manifest"),
        geometry_manifest_sha256=locks["geometry_manifest"]["sha256"],
        geometry=path("geometry"),
        geometry_sha256=locks["geometry"]["sha256"],
    )


def test_entrypoint_hash_constants_match_refrozen_preexecution_authority() -> None:
    assert sha256_path(MANIFEST) == EXPECTED_MANIFEST_SHA256
    assert sha256_path(PROTOCOL) == EXPECTED_PROTOCOL_SHA256
    record = json.loads(MANIFEST.read_text())
    validate_authority(record, _authority_args(record))


def test_entrypoint_has_no_checkpoint_model_or_inference_route() -> None:
    source = (ROOT / "paper0/tools/localize_b5_covariance.py").read_text(
        encoding="utf-8"
    )
    assert "import torch" not in source
    assert "load_selected" not in source
    assert "model.predict(" not in source
    assert "--checkpoint" not in source
    assert '"checkpoint_loaded": False' in source
    assert '"model_inference_performed": False' in source
    assert '"model_training_performed": False' in source
    assert '"held_out_85606_read": False' in source


def test_frozen_block_mapping_is_exhaustive_and_ordered() -> None:
    assert [_block_index(target) for target in range(498, 624)] == [
        index for index in range(6) for _ in range(21)
    ]
    assert _block_name(499, 519) == "frames_499_518"


def test_legacy_training_cross_check_requires_numerical_reproduction() -> None:
    spatial = {
        axis: {
            "fields": {
                field: {"correlation": [1.0, 0.5, 0.1]}
                for field in ("Ne", "Pe", "Pi", "phi", "Vi")
            }
        }
        for axis in ("x", "y", "stored_toroidal_z")
    }
    matrix = np.eye(5).tolist()
    record = {
        "spatial_autocorrelation": spatial,
        "cross_field": {
            "global": {"correlation_matrix": matrix},
            "eligible_union": {"correlation_matrix": matrix},
        },
    }
    bias = np.zeros((5, 4, 3))
    result = _legacy_training_cross_check(record, record, bias, bias)
    assert result["passed"] is True
    assert result["legacy_statistics_used_as_phi_gauge_fixed_reference"] is False

    discrepant = json.loads(json.dumps(record))
    discrepant["spatial_autocorrelation"]["x"]["fields"]["Ne"]["correlation"][1] = 0.4
    with pytest.raises(
        RuntimeError,
        match=r"spatial_max=0\.099.* at x/Ne; .*frozen_tolerance=",
    ):
        _legacy_training_cross_check(discrepant, record, bias, bias)


def test_legacy_training_batch_record_uses_frozen_direct_estimators() -> None:
    generator = np.random.default_rng(512)
    fluctuation = generator.normal(size=(3, 5, 4, 3, 8)).astype(np.float32)
    masks = {
        "eligible_union": np.ones((4, 3), dtype=bool),
        "left": np.asarray(
            [[True, True, True], [True, True, True], [False] * 3, [False] * 3]
        ),
    }
    record = _legacy_training_batch_record(fluctuation, masks)
    assert tuple(record["spatial_autocorrelation"]) == (
        "x",
        "y",
        "stored_toroidal_z",
    )
    assert tuple(record["cross_field"]) == ("global", "eligible_union", "left")
    result = _legacy_training_cross_check(
        record,
        record,
        np.zeros((5, 4, 3)),
        np.zeros((5, 4, 3)),
    )
    assert result["verification_estimator"] == (
        "exact_legacy_full_tensor_direct_dot_product"
    )
