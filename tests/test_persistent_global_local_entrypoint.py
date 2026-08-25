from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from paper0.tools.train_persistent_global_local_pilot import (
    authorize_manifest,
    exact_model_config,
    exact_noise_config,
)
from tcv_diagnostics.codec_training import sha256_path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "paper0/manifests/post_ecrd_old_85604_persistent_global_local_smoke.json"
)
PILOT_MANIFEST = (
    ROOT
    / "paper0/manifests/post_ecrd_old_85604_persistent_global_local_pilot.json"
)
PROTOCOL = (
    ROOT
    / "paper0/protocol/POST_ECRD_OLD_85604_PERSISTENT_GLOBAL_LOCAL_PILOT_2026-08-25.md"
)


def load_manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_smoke_manifest_authorizes_exact_configuration() -> None:
    manifest = load_manifest()
    authorize_manifest(
        manifest,
        mode="smoke",
        seed=1702,
        artifact_root=Path(str(manifest["artifact_root"])),
    )
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["new_nersc_data_access_allowed"] is False
    assert manifest["physics_derived_loss_allowed"] is False
    assert manifest["architecture"]["stochastic_parameter_count"] == 774234
    assert exact_model_config().low_mode_maximum == 7
    assert exact_noise_config().global_weight == 1.0


def test_manifest_locks_protocol_and_parent_records_by_sha256() -> None:
    manifest = load_manifest()
    assert manifest["protocol"]["sha256"] == sha256_path(PROTOCOL)
    source_manifest = ROOT / "paper0/manifests/post_ecrd_old_85604_four_step_feedback_pilot.json"
    parent_result = ROOT / "paper0/results/post_ecrd_old_85604_four_step_feedback_pilot_6937357.json"
    assert manifest["parent"]["source_manifest"]["sha256"] == sha256_path(source_manifest)
    assert manifest["parent"]["result"]["sha256"] == sha256_path(parent_result)
    assert manifest["parent"]["checkpoint"]["sha256"] == (
        "affe2589f4ce6639879ca1ed4a100af764aa48a475a653987faa18d4ce844117"
    )


def test_pilot_manifest_authorizes_full_budget_and_locks_smoke_scale() -> None:
    manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    authorize_manifest(
        manifest,
        mode="pilot",
        seed=1702,
        artifact_root=Path(str(manifest["artifact_root"])),
    )
    assert manifest["training"]["epochs"] == 20
    assert manifest["training"]["expected_optimizer_updates"] == 4280
    assert manifest["residual_scales"]["refit_in_pilot"] is False
    assert manifest["residual_scales"]["artifact"]["sha256"] == (
        "497a655bc6914c30d78831b04b157ad4c07e17a7de6c5887e378a36a53d475bd"
    )
    execution = ROOT / "paper0/protocol/POST_ECRD_OLD_85604_PERSISTENT_GLOBAL_LOCAL_PILOT_EXECUTION_AMENDMENT_2026-08-25.md"
    smoke = ROOT / "paper0/results/post_ecrd_old_85604_persistent_global_local_smoke_6937573.json"
    assert manifest["execution_amendment"]["sha256"] == sha256_path(execution)
    assert manifest["smoke_result"]["sha256"] == sha256_path(smoke)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("architecture", "low_mode_maximum"), 6),
        (("training", "expected_optimizer_updates"), 5),
        (("split", "zperiod"), 1),
        (("held_out_85606_access_allowed",), True),
        (("new_nersc_data_access_allowed",), True),
    ),
)
def test_manifest_tampering_is_rejected(path: tuple[str, ...], value: object) -> None:
    manifest = copy.deepcopy(load_manifest())
    target = manifest
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        authorize_manifest(
            manifest,
            mode="smoke",
            seed=1702,
            artifact_root=Path(str(manifest["artifact_root"])),
        )


def test_entrypoint_requires_online_wandb_and_keeps_scientific_scope_closed() -> None:
    source = (
        ROOT / "paper0/tools/train_persistent_global_local_pilot.py"
    ).read_text(encoding="utf-8")
    assert 'mode="online"' in source
    assert "verify_finished_wandb_run" in source
    assert '"held_out_85606_read": False' in source
    assert '"new_nersc_data_read": False' in source
    assert '"physics_derived_loss_used": False' in source
    assert "AutoregressiveStateWindowDataset" in source
