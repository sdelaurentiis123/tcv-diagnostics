"""Known-answer tests for the frozen B4 H-det/H-prob reducer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from tcv_diagnostics import pde_refiner_acceptance_gate as gate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "paper0/manifests/phase3_b4_full_evaluation_85604.json").read_text()
)


def _check(name: str, passes: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "kind": "numeric",
        "value": 1.0,
        "finite": True,
        "passes": passes,
    }


def _combined_family(
    det_name: str,
    prob_name: str,
    *,
    det_failed_blocks: tuple[int, ...] = (),
    prob_failed_blocks: tuple[int, ...] = (),
) -> dict[str, object]:
    blocks = []
    for index in range(6):
        checks = [
            _check(det_name.replace("overall", f"block{index}"), index not in det_failed_blocks),
            _check(prob_name.replace("overall", f"block{index}"), index not in prob_failed_blocks),
        ]
        blocks.append(
            {
                "passes": all(item["passes"] for item in checks),
                "checks": checks,
            }
        )
    return {
        "passes": True,
        "checks": [_check(det_name), _check(prob_name)],
        "chronological_blocks": blocks,
    }


def test_B4_gate_adapter_preserves_frozen_thresholds() -> None:
    adapted = gate.adapt_b4_numerical_gates(MANIFEST["gates"])
    assert adapted["field"][
        "aggregate_mean_mae_relative_to_paired_deterministic_max"
    ] == MANIFEST["gates"]["H_det"][
        "aggregate_mean_MAE_relative_to_parent_H1_max"
    ]
    assert adapted["spectral"][
        "material_calibration_spread_skill_range"
    ] == MANIFEST["gates"]["H_prob"][
        "material_power_and_cross_projection_spread_skill_range"
    ]
    assert adapted["transport"]["separatrix"][
        "fair_crps_better_than_paired_deterministic_required"
    ] == MANIFEST["gates"]["H_prob"][
        "separatrix_fair_CRPS_better_than_parent_H1_required"
    ]


def test_B4_projection_separates_deterministic_and_probabilistic_checks(
    monkeypatch,
) -> None:
    field = _combined_family(
        "field.overall.aggregate_mae_relative_to_paired_H2",
        "field.overall.Ne.spread_skill_relaxed",
    )
    spectral = _combined_family(
        "spectral.overall.field.Ne.k1_3.power_ratio",
        "spectral.overall.field.Ne.k1_3.calibration.spread_skill",
    )
    transport = _combined_family(
        "transport.overall.particle.strict_relative_l2",
        "transport.overall.particle.separatrix_fCRPS_relative_to_paired_H2_AE_relaxed",
    )
    monkeypatch.setattr(gate, "evaluate_field_family", lambda *_: field)
    monkeypatch.setattr(gate, "evaluate_spectral_family", lambda *_: spectral)
    monkeypatch.setattr(
        gate, "evaluate_transport_family_event_eligible", lambda *_: transport
    )
    score = {
        "transport_event_thresholds": {"spectral_materiality": {}},
        "field_and_marginal_calibration": {},
        "spectral_and_cross_field": {},
        "memberwise_transport": {},
    }
    comparator = {
        "field": {},
        "best_uncompressed": {"field": {}},
        "transport": {},
    }
    families = gate.evaluate_b4_numerical_families(
        score=score, comparator=comparator, manifest=MANIFEST
    )
    det_names = {
        item["name"]
        for family in families["H_det"].values()
        for item in family["checks"]
    }
    prob_names = {
        item["name"]
        for family in families["H_prob"].values()
        for item in family["checks"]
    }
    assert det_names == {
        "field.overall.aggregate_mae_relative_to_paired_H2",
        "spectral.overall.field.Ne.k1_3.power_ratio",
        "transport.overall.particle.strict_relative_l2",
    }
    assert prob_names == {
        "field.overall.Ne.spread_skill_relaxed",
        "spectral.overall.field.Ne.k1_3.calibration.spread_skill",
        "transport.overall.particle.separatrix_fCRPS_relative_to_paired_H2_AE_relaxed",
    }
    assert det_names.isdisjoint(prob_names)


def _projected_family(*, failed_blocks: tuple[int, ...] = ()) -> dict[str, object]:
    blocks = [
        {"passes": index not in failed_blocks, "checks": []}
        for index in range(6)
    ]
    return {
        "passes": len(failed_blocks) <= 1,
        "passes_overall": True,
        "passes_temporally": len(failed_blocks) <= 1,
        "blocks_passing": 6 - len(failed_blocks),
        "blocks_required": 5,
        "chronological_blocks": blocks,
        "checks": [],
    }


def test_B4_hypothesis_requires_same_five_blocks_across_families() -> None:
    # Every family independently passes five or six blocks, but failures occur
    # in different blocks.  Only four blocks pass jointly, so the hypothesis
    # must fail.
    record = gate._hypothesis_record(
        name="H_prob",
        integrity={"passes": True},
        families={
            "field": _projected_family(failed_blocks=(0,)),
            "spectral": _projected_family(failed_blocks=(1,)),
            "transport": _projected_family(),
        },
        stage_score=None,
    )
    assert record["joint_blocks_passing"] == 4
    assert record["passes"] is False


def test_B4_H_det_requires_stage_repair_but_H_prob_does_not() -> None:
    families = {
        "field": _projected_family(),
        "spectral": _projected_family(),
        "transport": _projected_family(),
    }
    failed_stage = {
        "stagewise_repair": {"gate_evaluated": True, "passes": False}
    }
    h_det = gate._hypothesis_record(
        name="H_det",
        integrity={"passes": True},
        families=families,
        stage_score=failed_stage,
    )
    h_prob = gate._hypothesis_record(
        name="H_prob",
        integrity={"passes": True},
        families=families,
        stage_score=None,
    )
    assert h_det["passes"] is False
    assert h_prob["passes"] is True


def test_B4_projection_rejects_missing_chronological_blocks() -> None:
    record = _combined_family(
        "field.overall.aggregate_mae_relative_to_paired_H2",
        "field.overall.Ne.spread_skill_relaxed",
    )
    record["chronological_blocks"] = record["chronological_blocks"][:5]
    try:
        gate._project_family(record, gate._field_det)
    except ValueError as error:
        assert "six chronological blocks" in str(error)
    else:
        raise AssertionError("five-block record unexpectedly accepted")


def test_B4_decision_keeps_H_det_and_H_prob_independent(monkeypatch) -> None:
    base = {
        "field": _projected_family(),
        "spectral": _projected_family(),
        "transport": _projected_family(),
    }
    monkeypatch.setattr(gate, "evaluate_b4_integrity", lambda **_: {"passes": True})
    monkeypatch.setattr(
        gate,
        "evaluate_b4_numerical_families",
        lambda **_: {"H_det": deepcopy(base), "H_prob": deepcopy(base)},
    )
    inputs = {
        "result": {"seed": 1701},
        "score": {"model_seed": 1701},
        "stage_score": {
            "stagewise_repair": {"gate_evaluated": True, "passes": True}
        },
        "training": {},
        "generation": {},
        "comparator": {},
        "manifest": MANIFEST,
        "training_wandb": {},
        "evaluation_wandb": {},
    }
    accepted = gate.evaluate_b4_one_seed_acceptance(**inputs)
    assert accepted["H_det"]["passes"] is True
    assert accepted["H_prob"]["passes"] is True
    assert accepted["joint_H_det_H_prob_pass"] is True
    assert accepted["seed1702_1703_replication_protocol_may_be_written"] is True
    assert accepted["seed1702_1703_training_authorized"] is False
    assert accepted["O3_protocol_may_be_written"] is False
    assert accepted["O3_launch_allowed"] is False

    failed_stage = deepcopy(inputs)
    failed_stage["stage_score"]["stagewise_repair"]["passes"] = False
    rejected = gate.evaluate_b4_one_seed_acceptance(**failed_stage)
    assert rejected["H_det"]["passes"] is False
    assert rejected["H_prob"]["passes"] is True
    assert rejected["joint_H_det_H_prob_pass"] is False
    assert rejected["disposition"] == MANIFEST["decision_rule"][
        "H_prob_pass_H_det_fail"
    ]
