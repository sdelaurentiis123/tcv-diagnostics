#!/usr/bin/env python3
"""Validate and summarize one paired full Stage-1 seed execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from tcv_diagnostics.model_data import assert_development_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    assert_development_path(args.job_root)
    assert_development_path(args.output)
    families = {}
    for family in ("c5p", "e6b"):
        path = args.job_root / family / "result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("family") != family or int(result.get("seed")) != args.seed:
            raise RuntimeError(f"{family} result identity differs")
        if result.get("held_out_85606_read") is not False:
            raise RuntimeError(f"{family} held-out flag differs")
        if result.get("physics_derived_loss_used") is not False:
            raise RuntimeError(f"{family} physics-loss flag differs")
        best = min(
            result["history"],
            key=lambda record: record["validation"][
                "shared_field_mean_model_derivative_mse"
            ],
        )
        families[family] = {
            "status": result["status"],
            "training_gate_passed": result["training_gate"]["passed"],
            "best_epoch": best["epoch"],
            "best_shared_field_derivative_mse": best["validation"][
                "shared_field_mean_model_derivative_mse"
            ],
            "best_shared_field_persistence_relative_skill": best["validation"][
                "shared_field_persistence_relative_skill"
            ],
            "best_per_field": best["validation"]["per_field"],
        }
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "scope": "post_ecrd_old_85604_stage1_full_paired_seed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "physics_derived_loss_used": False,
            "seed": args.seed,
            "families": families,
            "both_training_gates_passed": all(
                record["training_gate_passed"] for record in families.values()
            ),
        },
    )
    print(json.dumps(families, sort_keys=True))


if __name__ == "__main__":
    main()
