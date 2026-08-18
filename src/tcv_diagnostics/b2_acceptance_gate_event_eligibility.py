"""Versioned A016 correction for truth-empty B2 event blocks.

The prospectively frozen evaluator in :mod:`b2_acceptance_gate` is deliberately
left unchanged.  This module reuses every original non-event check and changes
only the chronological event checks authorized by A016.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

from .b2_acceptance_gate import (
    _transport_scope_checks,
    evaluate_b2_architecture_acceptance,
    evaluate_b2_seed_acceptance,
)
from .codec_transport import TRANSPORT_QUANTITIES


EVENT_BLOCK_POLICY = "truth_event_count_eligible_v1"
MINIMUM_ELIGIBLE_BLOCKS = 5


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or not number.is_integer():
        return None
    return int(number)


def _integrity_check(name: str, value: Any, passes: bool) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "integrity",
        "value": value,
        "median_eligible": False,
        "passes": bool(passes),
    }


def _not_applicable(name: str, *, event_count: int) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "not_applicable",
        "value": None,
        "reason": "truth_event_count_is_zero",
        "evidence": {"validation_event_count": int(event_count)},
        "median_eligible": False,
        "passes": True,
    }


def _refresh_record(record: Mapping[str, Any], checks: list[dict[str, Any]]) -> dict:
    refreshed = dict(record)
    refreshed["checks"] = checks
    refreshed["check_count"] = len(checks)
    refreshed["failed_check_count"] = sum(not item["passes"] for item in checks)
    refreshed["passes"] = bool(checks) and all(item["passes"] for item in checks)
    return refreshed


def _rewrite_chronological_event_checks(
    record: Mapping[str, Any],
    score: Mapping[str, Any],
    *,
    block_index: int,
) -> dict[str, Any]:
    """Replace only impossible zero-truth-event checks with explicit N/A."""

    checks = [dict(item) for item in record["checks"]]
    for quantity in TRANSPORT_QUANTITIES:
        prefix = f"transport.block{block_index}.{quantity}"
        event_names = {
            f"{prefix}.event_defined",
            f"{prefix}.event_magnitude_relative_error",
            f"{prefix}.event_sign_disagreement",
        }
        present = {item["name"] for item in checks if item["name"] in event_names}
        if present != event_names:
            raise ValueError(
                f"original B2 event-check schema differs for block {block_index} "
                f"quantity {quantity}"
            )
        event = score["quantities"][quantity]["upper_decile_event_conditioned"]
        count = _nonnegative_integer(event.get("validation_event_count"))
        checks.append(
            _integrity_check(
                f"{prefix}.event_count_is_nonnegative_integer",
                event.get("validation_event_count"),
                count is not None,
            )
        )
        if count != 0:
            continue

        checks = [item for item in checks if item["name"] not in event_names]
        checks.extend(
            [
                _integrity_check(
                    f"{prefix}.zero_event_record_is_undefined",
                    event.get("defined"),
                    event.get("defined") is False,
                ),
                _integrity_check(
                    f"{prefix}.zero_event_metrics_are_null",
                    {
                        "magnitude_relative_error": event.get(
                            "magnitude_relative_error"
                        ),
                        "truth_magnitude_weighted_sign_disagreement": event.get(
                            "truth_magnitude_weighted_sign_disagreement"
                        ),
                    },
                    event.get("magnitude_relative_error") is None
                    and event.get("truth_magnitude_weighted_sign_disagreement")
                    is None,
                ),
                _not_applicable(
                    f"{prefix}.event_conditioned_accuracy", event_count=count
                ),
            ]
        )
    return _refresh_record(record, checks)


def _event_checks_pass(block: Mapping[str, Any], *, quantity: str, index: int) -> bool:
    checks = {item["name"]: item for item in block["checks"]}
    prefix = f"transport.block{index}.{quantity}"
    names = (
        f"{prefix}.event_defined",
        f"{prefix}.event_magnitude_relative_error",
        f"{prefix}.event_sign_disagreement",
    )
    if any(name not in checks for name in names):
        raise ValueError(f"eligible B2 event-check schema differs for {prefix}")
    return all(checks[name]["passes"] for name in names)


def evaluate_transport_family_event_eligible(
    score: Mapping[str, Any],
    comparator: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the unchanged B2 transport gate with A016 block eligibility."""

    overall = _transport_scope_checks(
        score["overall"],
        comparator["overall"],
        gates,
        prefix="transport.overall",
        median_eligible=True,
        include_mc=True,
    )
    blocks = []
    for index, (block_score, block_comparator) in enumerate(
        zip(score["chronological_blocks"], comparator["chronological_blocks"])
    ):
        original = _transport_scope_checks(
            block_score,
            block_comparator,
            gates,
            prefix=f"transport.block{index}",
            median_eligible=False,
            include_mc=False,
        ).record()
        blocks.append(
            _rewrite_chronological_event_checks(
                original, block_score, block_index=index
            )
        )
    if len(blocks) != 6:
        raise ValueError("A016 requires the six frozen B2 chronological blocks")

    eligibility = {}
    all_event_counts_valid = True
    all_zero_event_records_consistent = True
    for quantity in TRANSPORT_QUANTITIES:
        eligible_indices = []
        eligible_passes = []
        for index, block_score in enumerate(score["chronological_blocks"]):
            event = block_score["quantities"][quantity][
                "upper_decile_event_conditioned"
            ]
            count = _nonnegative_integer(event.get("validation_event_count"))
            all_event_counts_valid = all_event_counts_valid and count is not None
            if count == 0:
                all_zero_event_records_consistent = bool(
                    all_zero_event_records_consistent
                    and event.get("defined") is False
                    and event.get("magnitude_relative_error") is None
                    and event.get("truth_magnitude_weighted_sign_disagreement")
                    is None
                )
            if count is not None and count > 0:
                eligible_indices.append(index)
                eligible_passes.append(
                    _event_checks_pass(blocks[index], quantity=quantity, index=index)
                )
        all_eligible_pass = bool(eligible_indices) and all(eligible_passes)
        eligibility[quantity] = {
            "eligible_block_indices": eligible_indices,
            "eligible_block_count": len(eligible_indices),
            "required_eligible_block_count": MINIMUM_ELIGIBLE_BLOCKS,
            "all_eligible_event_metrics_pass": all_eligible_pass,
        }
        overall.ge(
            f"transport.event_eligibility.{quantity}.eligible_block_count",
            len(eligible_indices),
            float(MINIMUM_ELIGIBLE_BLOCKS),
            median_eligible=False,
        )
        overall.boolean(
            f"transport.event_eligibility.{quantity}.all_eligible_metrics_pass",
            all_eligible_pass,
        )
    overall.boolean(
        "transport.event_eligibility.all_event_counts_valid",
        all_event_counts_valid,
    )
    overall.boolean(
        "transport.event_eligibility.all_zero_event_records_consistent",
        all_zero_event_records_consistent,
    )

    block_count = sum(item["passes"] for item in blocks)
    overall.ge("transport.blocks_passing", block_count, 5.0)
    result = overall.record()
    result.update(
        {
            "event_block_policy": EVENT_BLOCK_POLICY,
            "event_eligibility_by_quantity": eligibility,
            "chronological_blocks": blocks,
            "blocks_passing": block_count,
            "blocks_required": 5,
        }
    )
    return result


def evaluate_b2_seed_acceptance_event_eligible(
    *,
    result: Mapping[str, Any],
    score: Mapping[str, Any],
    training_run: Mapping[str, Any],
    comparator_run: Mapping[str, Any],
    best_uncompressed: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Reapply a seed gate while preserving every non-A016 original check."""

    amended = deepcopy(
        evaluate_b2_seed_acceptance(
            result=result,
            score=score,
            training_run=training_run,
            comparator_run=comparator_run,
            best_uncompressed=best_uncompressed,
            manifest=manifest,
        )
    )
    transport = evaluate_transport_family_event_eligible(
        score["memberwise_transport"],
        comparator_run["transport"],
        manifest["gates"]["transport"],
    )
    amended["families"]["transport"] = transport
    families = amended["families"]
    amended["numeric_checks_for_architecture_median"] = {
        item["name"]: item
        for family in families.values()
        for item in family["checks"]
        if item["kind"] == "numeric" and item["median_eligible"]
    }
    required_numeric = [
        item
        for family in families.values()
        for item in (
            list(family["checks"])
            + [
                check
                for block in family["chronological_blocks"]
                for check in block["checks"]
            ]
        )
        if item["kind"] == "numeric"
    ]
    all_numeric_finite = all(item["finite"] for item in required_numeric)
    event_record_integrity = all(
        item["passes"]
        for block in transport["chronological_blocks"]
        for item in block["checks"]
        if item["name"].endswith(
            (
                ".event_count_is_nonnegative_integer",
                ".zero_event_record_is_undefined",
                ".zero_event_metrics_are_null",
            )
        )
    )
    catastrophic = amended["catastrophic_bounds"]
    catastrophic["all_required_numeric_metrics_finite"] = all_numeric_finite
    catastrophic["event_block_record_integrity_passes"] = event_record_integrity
    catastrophic["passes"] = bool(
        catastrophic["integrity_passes"]
        and all_numeric_finite
        and event_record_integrity
        and catastrophic["ensemble_collapse_absent"]
        and catastrophic["aggregate_field_RMSE_and_MAE_within_1p20_paired_H2"]
        and catastrophic["all_separatrix_relative_l2_at_most_0p60"]
    )
    amended["passes_complete_per_seed_gate"] = bool(
        amended["integrity"]["passes"]
        and all(family["passes"] for family in families.values())
    )
    amended["event_block_policy"] = EVENT_BLOCK_POLICY
    amended["amendment"] = "A016"
    return amended


def evaluate_b2_architecture_acceptance_event_eligible(
    seed_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = tuple(seed_records)
    if any(item.get("event_block_policy") != EVENT_BLOCK_POLICY for item in records):
        raise ValueError("A016 architecture gate received a non-amended seed record")
    result = deepcopy(evaluate_b2_architecture_acceptance(records))
    result.update(
        {
            "schema_version": 2,
            "scope": (
                "phase3_B2_LDM_H2_A016_truth_event_eligible_"
                "architecture_acceptance_85604"
            ),
            "event_block_policy": EVENT_BLOCK_POLICY,
            "amendment": "A016",
            "original_frozen_gate_preserved": True,
            "outcome_informed_amendment": True,
        }
    )
    return result
