#!/usr/bin/env python3
"""Reduce the locked three-seed Stage-1 training and block evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


FAMILIES = ("c5p", "e6b")
SEEDS = (1701, 1702, 1703)
BLOCKS = ("V00", "V01", "V02")
E6B_FIELDS = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort")


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be finite and nonempty")
    return {
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def load_locked(record: dict[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    locked = record.get(key, {})
    path = Path(str(locked.get("path", "")))
    digest = str(locked.get("sha256", ""))
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{key} SHA-256 differs")
    return path, load_strict_json(path)


def best_training_record(result: dict[str, Any]) -> dict[str, Any]:
    history = result.get("history", [])
    if not history:
        raise ValueError("training result history is empty")
    best = min(
        history,
        key=lambda record: record["validation"][
            "shared_field_mean_model_derivative_mse"
        ],
    )
    expected = float(result["best_checkpoint"]["selection_metric"])
    actual = float(best["validation"]["shared_field_mean_model_derivative_mse"])
    if actual != expected:
        raise ValueError("training result best checkpoint metric differs")
    return best


def reduce_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("development_run") != "85604":
        raise ValueError("reducer manifest development run differs")
    if manifest.get("held_out_85606_read") is not False:
        raise ValueError("reducer manifest held-out flag differs")
    arms = manifest.get("arms", [])
    identities = {(str(arm.get("family")), int(arm.get("seed", -1))) for arm in arms}
    expected = {(family, seed) for family in FAMILIES for seed in SEEDS}
    if identities != expected or len(arms) != len(expected):
        raise ValueError("reducer manifest must lock exactly six Stage-1 arms")

    records: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in arms:
        family, seed = str(arm["family"]), int(arm["seed"])
        _, training = load_locked(arm, "training_result")
        _, blocked = load_locked(arm, "block_evaluation")
        for value in (training, blocked):
            if value.get("development_run") != "85604":
                raise ValueError("arm development run differs")
            if value.get("held_out_85606_read") is not False:
                raise ValueError("arm held-out flag differs")
            if value.get("physics_derived_loss_used") is not False:
                raise ValueError("arm physics-loss flag differs")
            if value.get("family") != family or int(value.get("seed")) != seed:
                raise ValueError("arm identity differs")
        if training.get("scope") != "post_ecrd_old_85604_stage1_codec_free_full":
            raise ValueError("training result scope differs")
        if blocked.get("scope") != (
            "post_ecrd_old_85604_stage1_chronological_block_evaluation"
        ):
            raise ValueError("block evaluation scope differs")
        if tuple(sorted(blocked.get("blocks", {}))) != BLOCKS:
            raise ValueError("block evaluation identities differ")
        records[(family, seed)] = {
            "training": training,
            "best": best_training_record(training),
            "blocks": blocked["blocks"],
        }

    families: dict[str, Any] = {}
    for family in FAMILIES:
        selected = [records[(family, seed)] for seed in SEEDS]
        fields = tuple(selected[0]["best"]["validation"]["per_field"])
        families[family] = {
            "seeds": list(SEEDS),
            "all_training_gates_passed": all(
                record["training"]["training_gate"]["passed"]
                for record in selected
            ),
            "best_epoch": numeric_summary(
                record["best"]["epoch"] for record in selected
            ),
            "shared_field_derivative_mse": numeric_summary(
                record["best"]["validation"][
                    "shared_field_mean_model_derivative_mse"
                ]
                for record in selected
            ),
            "shared_field_persistence_relative_skill": numeric_summary(
                record["best"]["validation"][
                    "shared_field_persistence_relative_skill"
                ]
                for record in selected
            ),
            "per_field_persistence_relative_skill": {
                field: numeric_summary(
                    record["best"]["validation"]["per_field"][field][
                        "persistence_relative_skill"
                    ]
                    for record in selected
                )
                for field in fields
            },
            "blocks": {},
        }
        for block in BLOCKS:
            families[family]["blocks"][block] = {
                "shared_field_derivative_mse": numeric_summary(
                    record["blocks"][block]["metrics"][
                        "shared_field_mean_model_derivative_mse"
                    ]
                    for record in selected
                ),
                "shared_field_persistence_relative_skill": numeric_summary(
                    record["blocks"][block]["metrics"][
                        "shared_field_persistence_relative_skill"
                    ]
                    for record in selected
                ),
                "per_field_persistence_relative_skill": {
                    field: numeric_summary(
                        record["blocks"][block]["metrics"]["per_field"][field][
                            "persistence_relative_skill"
                        ]
                        for record in selected
                    )
                    for field in fields
                },
            }

    c5p_mse = families["c5p"]["shared_field_derivative_mse"]["median"]
    e6b_mse = families["e6b"]["shared_field_derivative_mse"]["median"]
    aggregate_ratio = e6b_mse / c5p_mse
    block_ratios = {
        block: (
            families["e6b"]["blocks"][block]["shared_field_derivative_mse"][
                "median"
            ]
            / families["c5p"]["blocks"][block][
                "shared_field_derivative_mse"
            ]["median"]
        )
        for block in BLOCKS
    }
    evolved_overall_positive = {
        field: families["e6b"]["per_field_persistence_relative_skill"][field][
            "median"
        ]
        > 0.0
        for field in E6B_FIELDS
    }
    evolved_positive_blocks = {
        field: sum(
            families["e6b"]["blocks"][block][
                "per_field_persistence_relative_skill"
            ][field]["median"]
            > 0.0
            for block in BLOCKS
        )
        for field in E6B_FIELDS
    }
    gates = {
        "all_training_gates_passed": all(
            families[family]["all_training_gates_passed"] for family in FAMILIES
        ),
        "aggregate_e6b_to_c5p_mse_ratio_at_most_1_10": aggregate_ratio <= 1.10,
        "at_least_two_block_ratios_at_most_1_10": sum(
            ratio <= 1.10 for ratio in block_ratios.values()
        )
        >= 2,
        "all_block_ratios_at_most_1_25": all(
            ratio <= 1.25 for ratio in block_ratios.values()
        ),
        "every_e6b_field_positive_overall_skill": all(
            evolved_overall_positive.values()
        ),
        "every_e6b_field_positive_in_at_least_two_blocks": all(
            count >= 2 for count in evolved_positive_blocks.values()
        ),
    }
    advance_e6b = all(gates.values())
    return {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_stage1_full_reduction",
        "development_run": "85604",
        "held_out_85606_read": False,
        "physics_derived_loss_used": False,
        "families": families,
        "comparison": {
            "aggregate_e6b_to_c5p_mse_ratio": aggregate_ratio,
            "block_e6b_to_c5p_mse_ratios": block_ratios,
            "e6b_field_positive_overall_skill": evolved_overall_positive,
            "e6b_field_positive_block_counts": evolved_positive_blocks,
        },
        "decision_gates": gates,
        "decision": (
            "advance_e6b_as_primary_state"
            if advance_e6b
            else "retain_c5p_control_and_e6b_as_unresolved_exact_state_ablation"
        ),
        "advance_e6b_as_primary_state": advance_e6b,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.manifest)
    assert_development_path(args.output)
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("reducer manifest SHA-256 differs")
    reduced = reduce_manifest(load_strict_json(args.manifest))
    atomic_json(args.output, reduced)
    print(json.dumps(reduced, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
