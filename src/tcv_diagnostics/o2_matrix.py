"""Per-seed to per-arm decision logic for the frozen Paper 0 O2 matrix."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


O2_MATRIX_ORDER = (
    (0, "C5P-H1", 1701),
    (1, "C5P-H1", 1702),
    (2, "C5P-H1", 1703),
    (3, "C5P-H2", 1701),
    (4, "C5P-H2", 1702),
    (5, "C5P-H2", 1703),
)


def finalize_o2_matrix(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require all three seeds per arm; averaging cannot rescue failures."""

    records = list(runs)
    if len(records) != len(O2_MATRIX_ORDER):
        raise ValueError("O2 matrix requires exactly six run records")
    normalized = []
    for expected, run in zip(O2_MATRIX_ORDER, records):
        index, arm, seed = expected
        if (
            int(run.get("training_run_index", -1)) != index
            or run.get("arm") != arm
            or int(run.get("seed", -1)) != seed
        ):
            raise ValueError("O2 matrix run order or identity differs")
        if (
            run.get("scope") != "O2_selected_checkpoint_scientific_evaluation"
            or run.get("status") != "completed"
            or run.get("scientific_authority") is not True
            or run.get("development_run") != "85604"
            or run.get("held_out_85606_read") is not False
            or run.get("guard_frames_read") is not False
            or run.get("target_truth_used_during_forecast_generation") is not False
            or run.get("physics_derived_training_loss_used") is not False
            or run.get("O3_launch_allowed") is not False
        ):
            raise ValueError(f"O2 matrix run {index} provenance differs")
        gate = run.get("gate")
        if not isinstance(gate, Mapping):
            raise ValueError(f"O2 matrix run {index} gate is missing")
        gate_pass = gate.get("passes") is True and gate.get("status") == "pass"
        gate_fail = gate.get("passes") is False and gate.get("status") == "fail"
        if not (gate_pass or gate_fail):
            raise ValueError(f"O2 matrix run {index} gate status is inconsistent")
        if bool(run.get("O2_seed_accepted")) != gate_pass:
            raise ValueError(f"O2 matrix run {index} acceptance differs from gate")
        normalized.append(
            {
                "training_run_index": index,
                "arm": arm,
                "seed": seed,
                "passes": gate_pass,
                "status": "pass" if gate_pass else "fail",
            }
        )

    arms = {}
    accepted = []
    for arm in ("C5P-H1", "C5P-H2"):
        arm_runs = [record for record in normalized if record["arm"] == arm]
        passing = sum(record["passes"] for record in arm_runs)
        accepted_arm = passing == 3
        arms[arm] = {
            "required_seed_count": 3,
            "passing_seed_count": passing,
            "seed_results": {
                str(record["seed"]): record["status"] for record in arm_runs
            },
            "accepted": accepted_arm,
            "status": "pass" if accepted_arm else "fail",
        }
        if accepted_arm:
            accepted.append(arm)

    if not accepted:
        disposition = "stop_and_report_deterministic_one_step_failure"
        o3_protocol_may_be_frozen = False
    elif len(accepted) == 1:
        disposition = "sole_passing_arm_is_only_candidate_for_new_O3_protocol"
        o3_protocol_may_be_frozen = True
    else:
        disposition = "retain_both_arms_through_first_new_short_O3_comparison"
        o3_protocol_may_be_frozen = True
    return {
        "runs": normalized,
        "arms": arms,
        "accepted_arms": accepted,
        "accepted_arm_count": len(accepted),
        "seed_averaging_used": False,
        "all_three_seeds_required_per_arm": True,
        "disposition": disposition,
        "new_O3_protocol_may_be_frozen": o3_protocol_may_be_frozen,
        "O3_launch_allowed": False,
    }
