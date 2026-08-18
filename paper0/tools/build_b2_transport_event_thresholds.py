#!/usr/bin/env python3
"""Build the one frozen training-only B2 transport-event threshold artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_scoring import (  # noqa: E402
    compute_b2_transport_event_thresholds,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
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
    if actual != str(expected_sha256):
        raise ValueError(f"SHA-256 mismatch for {resolved}: {actual}")
    return resolved


def main() -> None:
    args = parse_args()
    for path in (
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
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
    geometry_path = verify_input(args.geometry, args.geometry_sha256)
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B2 thresholds {output}")
    output.mkdir(parents=True)
    native_truth = NativeTruthCatalog(load_strict_json(native_path))
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    thresholds = compute_b2_transport_event_thresholds(
        native_truth=native_truth,
        geometry=geometry,
    )
    thresholds.update(
        {
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "native_truth_result": {
                "path": str(native_path),
                "sha256": args.native_truth_result_sha256,
            },
            "geometry_manifest": {
                "path": str(geometry_manifest_path),
                "sha256": args.geometry_manifest_sha256,
            },
            "geometry": {
                "path": str(geometry_path),
                "sha256": args.geometry_sha256,
            },
        }
    )
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, thresholds)
    index_path = output / "artifact_sha256.txt"
    index_path.write_text(
        f"{sha256_path(result_path)}  {result_path.resolve(strict=True)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "result": str(result_path.resolve(strict=True)),
                "sha256": sha256_path(result_path),
                "held_out_85606_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
