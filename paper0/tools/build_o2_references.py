#!/usr/bin/env python3
"""Build and score frozen 85604 O2 references before checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.matched_o1_transport import (  # noqa: E402
    NativeTruthCatalog,
    load_transport_geometry,
)
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import load_official_catalog  # noqa: E402
from tcv_diagnostics.o2_forecast import O2ForecastArtifact  # noqa: E402
from tcv_diagnostics.o2_reference_evaluation import (  # noqa: E402
    O2_REFERENCE_NAMES,
    fit_training_only_o2_ar1,
    generate_o2_reference_forecast,
)
from tcv_diagnostics.o2_scoring import (  # noqa: E402
    compute_o2_training_materiality,
    score_o2_forecast,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"Paper 0 commit {actual} differs from {expected_commit}")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def verify_input(path: Path, expected_sha256: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    actual = sha256_path(resolved)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {resolved}: {actual}")
    return resolved


def _write_index(output: Path, artifacts: list[Path]) -> Path:
    index = output / "artifact_sha256.txt"
    if index.exists():
        raise FileExistsError(index)
    lines = [f"{sha256_path(path)}  {path.resolve(strict=True)}" for path in artifacts]
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main() -> None:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.evaluation_manifest,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    native_path = verify_input(
        args.native_truth_result, args.native_truth_result_sha256
    )
    geometry_manifest_path = verify_input(
        args.geometry_manifest, args.geometry_manifest_sha256
    )
    evaluation_manifest_path = verify_input(
        args.evaluation_manifest, args.evaluation_manifest_sha256
    )
    geometry_path = Path(args.geometry).resolve(strict=True)
    assert_development_path(geometry_path)
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite O2 references {output}")
    output.mkdir(parents=True)

    evaluation_manifest = load_strict_json(evaluation_manifest_path)
    if (
        evaluation_manifest.get("development_run") != "85604"
        or evaluation_manifest.get("held_out_85606_access_allowed") is not False
        or evaluation_manifest.get("status")
        != "frozen_before_O2_scientific_evaluation"
    ):
        raise RuntimeError("O2 evaluation manifest is not frozen for 85604")
    catalog = load_official_catalog(args.artifact_root)
    native_truth = NativeTruthCatalog(load_strict_json(native_path))
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    targets = tuple(range(498, 624 if args.mode == "full" else 502))
    scientific = args.mode == "full"

    materiality_path = output / "training_materiality.json"
    materiality = compute_o2_training_materiality(catalog)
    write_strict_json_atomic(materiality_path, materiality)
    ar1 = fit_training_only_o2_ar1(catalog)
    ar1_path = output / "training_spectral_ar1.json"
    ar1_record = {
        **ar1.to_record(),
        "development_run": "85604",
        "held_out_85606_read": False,
        "fit_frames": [0, 432],
        "fit_pairs": [[0, 1], [430, 431]],
        "Bphi_fit": "inapplicable_to_C5P_only_continuation",
    }
    write_strict_json_atomic(ar1_path, ar1_record)

    artifacts = [materiality_path, ar1_path]
    reference_records: dict[str, Any] = {}
    common_metadata = {
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "evaluation_manifest": {
            "path": str(evaluation_manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "mode": args.mode,
    }
    for name in O2_REFERENCE_NAMES:
        forecast_path = output / f"{name}_forecast.h5"
        generation = generate_o2_reference_forecast(
            catalog=catalog,
            name=name,
            target_frames=targets,
            output=forecast_path,
            spectral_ar1=ar1 if name == "spectral_ar1" else None,
            metadata=common_metadata,
        )
        generation_path = output / f"{name}_generation.json"
        write_strict_json_atomic(generation_path, generation)
        with O2ForecastArtifact(
            forecast_path,
            expected_sha256=generation["forecast"]["sha256"],
            target_frames=targets,
        ) as forecast:
            score = score_o2_forecast(
                catalog=catalog,
                forecast_artifact=forecast,
                native_truth=native_truth,
                geometry=geometry,
                target_frames=targets,
                scientific_authority=scientific,
            )
        score_path = output / f"{name}_score.json"
        write_strict_json_atomic(score_path, score)
        artifacts.extend((forecast_path, generation_path, score_path))
        reference_records[name] = {
            "generation": {
                "path": str(generation_path.resolve(strict=True)),
                "sha256": sha256_path(generation_path),
            },
            "forecast": {
                "path": str(forecast_path.resolve(strict=True)),
                "sha256": sha256_path(forecast_path),
            },
            "score": {
                "path": str(score_path.resolve(strict=True)),
                "sha256": sha256_path(score_path),
            },
        }

    result = {
        "schema_version": 1,
        "scope": "O2_frozen_uncompressed_references",
        "status": "completed",
        "scientific_authority": scientific,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "training_performed": False,
        "validation_tuning_used": False,
        "target_truth_read_during_forecast_generation": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "mode": args.mode,
        "target_frames": [targets[0], targets[-1] + 1],
        "training_materiality": {
            "path": str(materiality_path.resolve(strict=True)),
            "sha256": sha256_path(materiality_path),
        },
        "training_spectral_ar1": {
            "path": str(ar1_path.resolve(strict=True)),
            "sha256": sha256_path(ar1_path),
        },
        "references": reference_records,
        "evaluation_manifest": {
            "path": str(evaluation_manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "O2_seed_gate_evaluated": False,
        "O3_launch_allowed": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    artifacts.append(result_path)
    index = _write_index(output, artifacts)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sha256": sha256_path(result_path),
                "artifact_index": str(index),
                "artifact_index_sha256": sha256_path(index),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
