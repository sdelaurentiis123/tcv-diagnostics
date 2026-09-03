#!/usr/bin/env python3
"""Reduce all six hierarchical M32 scores with the frozen stopping rule."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import (
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.pgl_hierarchical_decision import evaluate_two_epoch_decision
from tcv_diagnostics.pgl_hierarchical_training import (
    PGL_HIERARCHICAL_ARMS,
    PGL_HIERARCHICAL_CHECKPOINT_UPDATES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def verify_checkout(root: Path, expected: str) -> None:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != expected or dirty:
        raise RuntimeError("hierarchical reduction requires the locked clean checkout")


def load_six_results(
    root: Path, *, commit: str
) -> tuple[dict, dict, list[dict]]:
    records = {}
    scores = {}
    artifacts = []
    task = 0
    for arm in PGL_HIERARCHICAL_ARMS:
        for update in PGL_HIERARCHICAL_CHECKPOINT_UPDATES:
            path = (
                root
                / f"task_{task}_{arm}_u{update:04d}"
                / "scoring"
                / "result.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            result = load_strict_json(path)
            if (
                result.get("scope")
                != "old_85604_pgl_hierarchical_physics_evaluation"
                or result.get("status") not in ("completed_passed", "completed_failed")
                or result.get("development_run") != "85604"
                or result.get("arm") != arm
                or result.get("optimizer_update") != update
                or result.get("paper0_commit") != commit
                or result.get("training_performed") is not False
                or result.get("checkpoint_selection_performed") is not False
                or result.get("target_truth_used_during_generation") is not False
                or result.get("physics_derived_training_loss_used") is not (
                    arm == "TRANSPORT"
                )
                or result.get("held_out_85606_read") is not False
                or result.get("new_nersc_data_read") is not False
            ):
                raise ValueError(f"hierarchical score contract differs for task {task}")
            score_path = Path(str(result["score"]["path"]))
            assert_development_path(score_path)
            if sha256_path(score_path) != result["score"]["sha256"]:
                raise ValueError(f"hierarchical score SHA differs for task {task}")
            score = load_strict_json(score_path)
            hierarchy = score.get("hierarchical_transport_evaluation", {})
            if (
                score.get("scope")
                != "old_85604_pgl_hierarchical_truth_separated_physics_scoring"
                or score.get("arm") != arm
                or score.get("optimizer_update") != update
                or hierarchy.get("scope")
                != "old_85604_pgl_hierarchical_validation_scores"
                or hierarchy.get("held_out_85606_read") is not False
                or hierarchy.get("new_nersc_data_read") is not False
            ):
                raise ValueError(f"hierarchical detailed score differs for task {task}")
            records[(arm, update)] = result
            scores[(arm, update)] = score
            artifacts.append(
                {
                    "task": task,
                    "arm": arm,
                    "optimizer_update": update,
                    "result": {"path": str(path), "sha256": sha256_path(path)},
                    "score": result["score"],
                }
            )
            task += 1
    return records, scores, artifacts


def _ratio(record: dict) -> float | None:
    value = record.get("spread_skill_ratio")
    return None if value is None else float(value)


def write_hierarchy_tables(scores: dict, output: Path) -> tuple[Path, Path]:
    """Write tidy tables for every score scale and variogram group."""

    hierarchy_path = output / "hierarchy_metrics.csv"
    names = (
        "arm",
        "optimizer_update",
        "equivalent_epochs",
        "quantity",
        "local_spatial_variogram",
        "local_temporal_variogram",
        "regional_energy",
        "fourier_low_n5_15_energy",
        "fourier_n20_35_energy",
        "global_crps",
        "regional_spread_skill",
        "fourier_low_n5_15_spread_skill",
        "fourier_n20_35_spread_skill",
        "global_n0_spread_skill",
        "regional_covariance_relative_frobenius_error",
        "fourier_low_n5_15_covariance_relative_frobenius_error",
        "fourier_n20_35_covariance_relative_frobenius_error",
    )
    with hierarchy_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for arm in PGL_HIERARCHICAL_ARMS:
            for update in PGL_HIERARCHICAL_CHECKPOINT_UPDATES:
                hierarchy = scores[(arm, update)]["hierarchical_transport_evaluation"]
                for quantity, record in hierarchy["quantities"].items():
                    fair = record["fair_scores"]
                    spread = record["spread_skill"]
                    covariance = record["covariance_match"]
                    writer.writerow(
                        {
                            "arm": arm,
                            "optimizer_update": update,
                            "equivalent_epochs": update / 214,
                            "quantity": quantity,
                            "local_spatial_variogram": fair[
                                "local_spatial_variogram"
                            ],
                            "local_temporal_variogram": fair[
                                "local_temporal_variogram"
                            ],
                            "regional_energy": fair["regional_energy"],
                            "fourier_low_n5_15_energy": fair[
                                "fourier_low_energy"
                            ],
                            "fourier_n20_35_energy": fair[
                                "fourier_n20_35_energy"
                            ],
                            "global_crps": fair["global_crps"],
                            "regional_spread_skill": _ratio(spread["regional"]),
                            "fourier_low_n5_15_spread_skill": _ratio(
                                spread["fourier_low_n5_15"]
                            ),
                            "fourier_n20_35_spread_skill": _ratio(
                                spread["fourier_n20_35"]
                            ),
                            "global_n0_spread_skill": _ratio(spread["global_n0"]),
                            "regional_covariance_relative_frobenius_error": covariance[
                                "regional_12_sector"
                            ]["relative_frobenius_error"],
                            "fourier_low_n5_15_covariance_relative_frobenius_error": covariance[
                                "fourier_low_n5_15"
                            ]["relative_frobenius_error"],
                            "fourier_n20_35_covariance_relative_frobenius_error": covariance[
                                "fourier_n20_35"
                            ]["relative_frobenius_error"],
                        }
                    )

    curve_path = output / "variogram_curves.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        names = (
            "arm",
            "optimizer_update",
            "equivalent_epochs",
            "quantity",
            "axis",
            "group_index",
            "group_value",
            "fair_variogram_score",
        )
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for arm in PGL_HIERARCHICAL_ARMS:
            for update in PGL_HIERARCHICAL_CHECKPOINT_UPDATES:
                hierarchy = scores[(arm, update)]["hierarchical_transport_evaluation"]
                groups = (
                    (
                        "physical_distance_m",
                        hierarchy["spatial_distance_bin_upper_edges_m"],
                        "spatial_variogram_by_distance_bin",
                    ),
                    (
                        "temporal_lag_microseconds",
                        hierarchy["temporal_lags_microseconds"],
                        "temporal_variogram_by_lag",
                    ),
                )
                for quantity, record in hierarchy["quantities"].items():
                    for axis, values, score_name in groups:
                        for index, (value, score) in enumerate(
                            zip(values, record[score_name], strict=True)
                        ):
                            writer.writerow(
                                {
                                    "arm": arm,
                                    "optimizer_update": update,
                                    "equivalent_epochs": update / 214,
                                    "quantity": quantity,
                                    "axis": axis,
                                    "group_index": index,
                                    "group_value": value,
                                    "fair_variogram_score": score,
                                }
                            )
    return hierarchy_path, curve_path


def write_decision_readout(decision: dict, output: Path) -> Path:
    """Write a concise, generated interpretation of the frozen decision."""

    control = decision["metrics"]["CONTROL_update_428"]
    treatment = decision["metrics"]["TRANSPORT_update_428"]
    delta = decision["epoch_two_matched_difference"]
    lines = [
        "# Hierarchical transport-loss screen\n",
        "Old 85604 only; matched two-epoch warm starts; M32 validation. "
        "The TRANSPORT arm is explicitly transport-supervised.\n",
        "| Two-epoch metric | Control | Transport-aware | Required |\n",
        "| --- | ---: | ---: | ---: |\n",
        f"| Integrated spread--skill | {control['integrated_spread_skill']:.4f} | "
        f"{treatment['integrated_spread_skill']:.4f} | gain >= 0.05 |\n",
        f"| Spatial covariance error | {control['spatial_covariance_error']:.4f} | "
        f"{treatment['spatial_covariance_error']:.4f} | reduction >= 0.01 |\n",
        f"| Local spread--skill (median) | {control['local_spread_skill_median']:.4f} | "
        f"{treatment['local_spread_skill_median']:.4f} | production gate |\n",
        f"| Mean transport relative L2 | {control['mean_transport_relative_l2']:.4f} | "
        f"{treatment['mean_transport_relative_l2']:.4f} | must remain stable |\n",
        "\n",
        f"Integrated spread gain: `{delta['integrated_spread_skill_gain']:.4f}`.  \n",
        f"Covariance-error reduction: `{delta['spatial_covariance_error_reduction']:.4f}`.  \n",
        f"Decision: **{decision['next_action']}**.\n",
        "\nNo 85606 or new NERSC data were read.\n",
    ]
    path = output / "DECISION_READOUT.md"
    path.write_text("".join(lines), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    for path in (args.score_root, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_checkout(args.paper0_root, args.paper0_commit)
    records, scores, artifacts = load_six_results(
        args.score_root, commit=args.paper0_commit
    )
    decision = evaluate_two_epoch_decision(records)
    decision.update(
        {
            "status": "completed",
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "inputs": artifacts,
            "checkpoint_selection_performed": False,
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
        }
    )
    args.output.mkdir(parents=True)
    decision_path = args.output / "decision.json"
    write_strict_json_atomic(decision_path, decision)
    csv_path = args.output / "checkpoint_metrics.csv"
    names = (
        "arm",
        "optimizer_update",
        "equivalent_epochs",
        "integrated_spread_skill",
        "spatial_covariance_error",
        "local_spread_skill_median",
        "mean_transport_relative_l2",
        "spectral_error",
        "cross_spectrum_error",
        "phase_error_degrees",
        "field_fair_crps_h4",
        "production_passed",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for arm in PGL_HIERARCHICAL_ARMS:
            for update in PGL_HIERARCHICAL_CHECKPOINT_UPDATES:
                metric = decision["metrics"][f"{arm}_update_{update}"]
                writer.writerow(
                    {
                        "arm": arm,
                        "optimizer_update": update,
                        "equivalent_epochs": update / 214,
                        **{name: metric[name] for name in names[3:]},
                    }
                )
    hierarchy_path, curve_path = write_hierarchy_tables(scores, args.output)
    readout_path = write_decision_readout(decision, args.output)
    manifest = {
        "schema_version": 1,
        "scope": "old_85604_pgl_hierarchical_screen_reduction_manifest",
        "status": "completed",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "score_root": str(args.score_root),
        "inputs": artifacts,
        "outputs": {
            "decision": {
                "path": str(decision_path),
                "sha256": sha256_path(decision_path),
            },
            "metrics_csv": {
                "path": str(csv_path),
                "sha256": sha256_path(csv_path),
            },
            "hierarchy_metrics_csv": {
                "path": str(hierarchy_path),
                "sha256": sha256_path(hierarchy_path),
            },
            "variogram_curves_csv": {
                "path": str(curve_path),
                "sha256": sha256_path(curve_path),
            },
            "decision_readout": {
                "path": str(readout_path),
                "sha256": sha256_path(readout_path),
            },
        },
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    write_strict_json_atomic(args.output / "manifest.json", manifest)
    print(json.dumps({"decision": decision, "manifest": manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
