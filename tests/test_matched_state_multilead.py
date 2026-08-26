"""Known-answer checks for the old-85604 matched state-view pilot."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from paper0.tools.train_matched_state_multilead import (
    LEADS,
    authorize_manifest,
    build_model,
    transition_gate,
)
from tcv_diagnostics.state_operator_data import plan_lead_pairs


MANIFEST = Path(
    "paper0/manifests/post_ecrd_old_85604_matched_state_multilead.json"
)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _validation(family: str, *, skill: float = 0.2) -> dict:
    fields = (
        ("Ne", "Pe", "Pi", "phi", "Vi")
        if family == "c5p"
        else ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort")
    )
    per_lead = {}
    for lead in LEADS:
        record = {
            "per_field": {
                field: {"persistence_relative_skill": skill}
                for field in fields
            }
        }
        if family == "e6b":
            record["boundary_by_side"] = {
                side: {"persistence_relative_skill": skill}
                for side in ("inner", "outer")
            }
        per_lead[str(lead)] = record
    return {"per_lead": per_lead}


def test_manifest_authorizes_only_the_matched_old_85604_pair() -> None:
    manifest = _manifest()
    authorize_manifest(manifest, family="c5p", seed=1701)
    authorize_manifest(manifest, family="e6b", seed=1701)

    for forbidden_key in (
        "held_out_85606_access_allowed",
        "new_nersc_data_access_allowed",
        "guard_frames_read_allowed",
    ):
        broken = copy.deepcopy(manifest)
        broken[forbidden_key] = True
        with pytest.raises(ValueError, match="scope flags"):
            authorize_manifest(broken, family="c5p", seed=1701)

    with pytest.raises(ValueError, match="seed"):
        authorize_manifest(manifest, family="e6b", seed=1702)


def test_manifest_locks_duration_and_pair_counts() -> None:
    manifest = _manifest()
    broken = copy.deepcopy(manifest)
    broken["optimization"]["epochs"] = 20
    with pytest.raises(ValueError, match="optimization"):
        authorize_manifest(broken, family="c5p", seed=1701)

    train = plan_lead_pairs(split="train", lead_steps=LEADS, history_frames=1)
    validation = plan_lead_pairs(
        split="validation", lead_steps=LEADS, history_frames=1
    )
    assert len(train) == 2129
    assert len(validation) == 609
    consumed = {
        frame
        for pair in train + validation
        for frame in (pair.current, pair.target)
    }
    assert not any(432 <= frame < 496 for frame in consumed)


def test_matched_processors_preserve_toroidal_resolution() -> None:
    architecture = _manifest()["architecture"]
    c5p, c5p_config = build_model(architecture, family="c5p")
    e6b, e6b_config = build_model(architecture, family="e6b")

    assert c5p_config.predict_boundary is False
    assert e6b_config.predict_boundary is True
    assert c5p_config.to_record()["downsample_stride_xyz"] == [2, 2, 1]
    assert e6b_config.to_record()["downsample_stride_xyz"] == [2, 2, 1]
    assert c5p.to_record()["parameter_count"] == 2174021
    assert e6b.to_record()["parameter_count"] == 2181704


def test_transition_gate_requires_all_evolved_and_boundary_skills() -> None:
    assert transition_gate(_validation("c5p"), family="c5p")["passed"]
    assert transition_gate(_validation("e6b"), family="e6b")["passed"]

    failed_volume = _validation("e6b")
    failed_volume["per_lead"]["16"]["per_field"]["Vort"][
        "persistence_relative_skill"
    ] = -0.01
    assert not transition_gate(failed_volume, family="e6b")["passed"]

    failed_boundary = _validation("e6b")
    failed_boundary["per_lead"]["8"]["boundary_by_side"]["outer"][
        "persistence_relative_skill"
    ] = 0.0
    gate = transition_gate(failed_boundary, family="e6b")
    assert gate["every_volume_field_positive_skill_at_every_lead"]
    assert not gate["every_boundary_side_positive_skill_at_every_lead"]
    assert not gate["passed"]
