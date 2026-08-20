"""Fail-closed checks for the prospective ECRD development protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
MANIFEST = ROOT / "paper0/manifests/ecrd_model_development_85604.json"


def test_protocol_hash_and_sequestered_scope() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    observed = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert manifest["protocol"]["sha256"] == observed
    assert manifest["development_run"] == "85604"
    assert manifest["sequestered_run"] == "85606"
    assert manifest["held_out_85606_access_allowed"] is False
    assert "85606_access_before_explicit_release_record" in manifest["forbidden_scope"]
    assert "guard_reads" in manifest["forbidden_scope"]


def test_frozen_boundaries_and_arm_ladder() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data = manifest["data"]
    assert data["training_targets"] == [2, 432]
    assert data["guard_frames"] == [432, 496]
    assert data["validation_targets"] == [498, 624]
    assert data["validation_blocks"] == {
        "V00": [498, 540],
        "V01": [540, 582],
        "V02": [582, 624],
    }
    assert data["periodic_axes_xyz"] == [False, False, True]
    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5k"
    assert manifest["model_seeds"] == [1701, 1702, 1703]
    arms = {record["name"]: record for record in manifest["arms"]}
    assert tuple(arms) == ("B5", "B5-Context", "ECRD", "ECRD-History")
    assert arms["B5"]["deep_conditioning"] is False
    assert arms["B5-Context"]["deep_conditioning"] is True
    assert arms["B5-Context"]["toroidal_downsampling"] is True
    assert arms["ECRD"]["toroidal_downsampling"] is False
    assert arms["ECRD"]["mean_head"] is True
    assert arms["ECRD"]["multiscale_noise"] is True
    assert arms["ECRD-History"]["history_frames"] == 2


def test_training_is_field_only_and_downstream_is_closed() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["training"]["physics_derived_loss_allowed"] is False
    assert manifest["sampling"]["posthoc_inflation_allowed"] is False
    assert manifest["selection"]["physics_checkpoint_selection_allowed"] is False
    forbidden = set(manifest["forbidden_scope"])
    assert {"assimilation", "diagnostic_ranking", "steering_or_control"} <= forbidden


def test_model_module_has_no_physics_or_data_dependency() -> None:
    source = (ROOT / "src/tcv_diagnostics/models/ecrd.py").read_text(
        encoding="utf-8"
    )
    assert "from ..transport" not in source
    assert "from ..spect" not in source
    assert "h5py" not in source
    assert "85606" not in source
