"""Contract checks for bounded-rollout evaluation-only physics scoring."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from paper0.tools.score_codec_free_bounded_rollout_physics import (
    authorize_state_artifact,
    validate_forecast_schema,
)
from tcv_diagnostics.bounded_rollout import FIELDS, method_schedule
from tcv_diagnostics.codec_training import sha256_path


def _write_state_result(
    path: Path,
    *,
    forecast: Path,
    forecast_sha256: str,
    manifest: Path,
    manifest_sha256: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "scope": "post_ecrd_old_85604_bounded_rollout",
                "status": "bounded_state_forecast_generated_and_scored",
                "development_run": "85604",
                "held_out_85606_read": False,
                "new_nersc_data_read": False,
                "guard_frames_read": False,
                "training_performed": False,
                "checkpoint_selection_performed": False,
                "physics_derived_loss_used": False,
                "physics_diagnostics_scored": False,
                "physics_scoring_authorized_next": True,
                "paper0_commit": "a" * 40,
                "manifest": str(manifest),
                "manifest_sha256": manifest_sha256,
                "forecast_artifact": {
                    "path": str(forecast),
                    "sha256": forecast_sha256,
                    "stored_value": ("standardized_terminal_state_delta_from_current"),
                    "dtype": "float32",
                },
            }
        ),
        encoding="utf-8",
    )


def test_state_artifact_authorization_is_hash_and_scope_locked(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    manifest_sha256 = sha256_path(manifest)
    forecast = tmp_path / "forecast.h5"
    forecast.write_bytes(b"immutable forecast")
    forecast_sha256 = sha256_path(forecast)
    result = tmp_path / "result.json"
    _write_state_result(
        result,
        forecast=forecast,
        forecast_sha256=forecast_sha256,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    record = authorize_state_artifact(
        result_path=result,
        result_sha256=sha256_path(result),
        forecast_path=forecast,
        forecast_sha256=forecast_sha256,
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
    )
    assert record["paper0_commit"] == "a" * 40

    modified = json.loads(result.read_text(encoding="utf-8"))
    modified["physics_diagnostics_scored"] = True
    result.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="contract differs"):
        authorize_state_artifact(
            result_path=result,
            result_sha256=sha256_path(result),
            forecast_path=forecast,
            forecast_sha256=forecast_sha256,
            manifest_path=manifest,
            manifest_sha256=manifest_sha256,
        )


def test_forecast_schema_requires_every_frozen_seed_and_method(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forecast.h5"
    commit = "b" * 40
    manifest_sha256 = "c" * 64
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": 1,
                "scope": "post_ecrd_old_85604_bounded_rollout",
                "development_run": "85604",
                "held_out_85606_read": False,
                "new_nersc_data_read": False,
                "guard_frames_read": False,
                "zperiod": 5,
                "stored_value": ("standardized_terminal_state_delta_from_current"),
                "paper0_commit": commit,
                "manifest_sha256": manifest_sha256,
                "fields": json.dumps(list(FIELDS)),
            }
        )
        for horizon in (4, 8):
            count = 624 - horizon - 496
            current = np.arange(496, 624 - horizon, dtype=np.int64)
            group = handle.create_group(f"horizon_{horizon}")
            group.attrs["horizon_saved_frames"] = horizon
            group.attrs["pair_count"] = count
            group.create_dataset("current_frame", data=current)
            group.create_dataset("target_frame", data=current + horizon)
            for seed in (1701, 1702, 1703):
                seed_group = group.create_group(f"seed_{seed}")
                for method in method_schedule(horizon):
                    seed_group.create_dataset(
                        method,
                        shape=(count, len(FIELDS), 64, 32, 88),
                        dtype="f4",
                        chunks=(1, 1, 16, 16, 22),
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                        fletcher32=True,
                    )
        validate_forecast_schema(
            handle,
            paper0_commit=commit,
            manifest_sha256=manifest_sha256,
        )
        del handle["horizon_8/seed_1703/autoregressive_lead4"]
        with pytest.raises(ValueError, match="method inventory differs"):
            validate_forecast_schema(
                handle,
                paper0_commit=commit,
                manifest_sha256=manifest_sha256,
            )
