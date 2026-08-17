#!/usr/bin/env python3
"""Evaluate one selected O1 codec and publish locked native-81 candidates.

This is the first half of the matched O1 evaluation.  It computes native-view
field/spectral metrics and writes the exact codec reconstruction inputs needed
by the separately compiled elliptic and transport oracle.  It never reads
85606 and it never substitutes an approximate potential solve.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.matched_codec_evaluation import (
    NATIVE81_FIELDS,
    decode_physical_batch,
    native81_candidate_fields,
    native_view_spec,
    restore_selected_codec,
    validate_selected_checkpoint,
)
from tcv_diagnostics.matched_codec_metrics import (
    MatchedCodecAccumulator,
    build_matched_o1_view_gate,
    training_materiality,
)
from tcv_diagnostics.model_data import (
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import (
    CodecFrameDataset,
    FAMILY_FIELDS,
    OFFICIAL_ARTIFACT_ROOT,
    TRAIN_INTERVAL,
    VALIDATION_INTERVAL,
    VOLUME_SHAPE,
    load_official_catalog,
)

NATIVE_SHAPE = (64, 32, 81)
VALIDATION_BLOCK_FRAMES = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", choices=("dcae_l20", "dcae_l10"), required=True)
    parser.add_argument("--family", choices=("c5p", "e6b"), required=True)
    parser.add_argument("--seed", type=int, choices=(1701, 1702, 1703), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--chunk-frames", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"Paper 0 commit mismatch: {actual} != {expected_commit}")
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


def configure_determinism(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "seed": 0,
        "inference_dtype": "float32",
        "metric_accumulator_dtype": "float64",
        "tf32_allowed": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
    }


class CandidateWriter:
    """Write one immutable downstream candidate HDF5 artifact atomically."""

    def __init__(
        self,
        path: Path,
        *,
        family: str,
        frames: tuple[int, ...],
        codec: str,
        seed: int,
        checkpoint_sha256: str,
    ) -> None:
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        if path.exists() or self.temporary.exists():
            raise FileExistsError(path)
        self.handle = h5py.File(self.temporary, "x")
        self.handle.attrs["schema_version"] = 1
        self.handle.attrs["development_run"] = "85604"
        self.handle.attrs["held_out_85606_read"] = False
        self.handle.attrs["family"] = family
        self.handle.attrs["codec"] = codec
        self.handle.attrs["seed"] = int(seed)
        self.handle.attrs["checkpoint_sha256"] = checkpoint_sha256
        self.handle.attrs["zperiod"] = 5
        self.handle.attrs["native_shape"] = NATIVE_SHAPE
        coordinates = self.handle.create_group("coordinates")
        coordinates.create_dataset(
            "frame_index", data=np.asarray(frames, dtype=np.int64)
        )
        candidate = self.handle.create_group("candidate")
        self.fields = {
            field: candidate.create_dataset(
                field,
                shape=(len(frames), *NATIVE_SHAPE),
                dtype="f4",
                chunks=(1, *NATIVE_SHAPE),
                compression="lzf",
            )
            for field in NATIVE81_FIELDS[family]
        }
        self.boundary = None
        if family == "e6b":
            boundary = self.handle.create_group("boundary")
            self.boundary = boundary.create_dataset(
                "Bphi",
                shape=(len(frames), 2, 32),
                dtype="f4",
                chunks=(1, 2, 32),
            )
            self.boundary.attrs["policy"] = "exact_bypass_from_model_dataset"
        self.written = np.zeros(len(frames), dtype=bool)

    def write(
        self,
        start: int,
        values: dict[str, np.ndarray],
        boundary: np.ndarray | None,
    ) -> None:
        count = next(iter(values.values())).shape[0]
        stop = start + count
        if start < 0 or stop > self.written.size or np.any(self.written[start:stop]):
            raise ValueError("candidate write is overlapping or out of range")
        if set(values) != set(self.fields):
            raise ValueError("candidate field set differs from the family contract")
        for field, array in values.items():
            data = np.asarray(array, dtype=np.float32)
            if data.shape != (count, *NATIVE_SHAPE) or not np.all(np.isfinite(data)):
                raise ValueError(f"invalid native candidate field {field}")
            self.fields[field][start:stop] = data
        if self.boundary is None:
            if boundary is not None:
                raise ValueError("C5P candidate cannot contain a boundary")
        else:
            data = np.asarray(boundary, dtype=np.float32)
            if data.shape != (count, 2, 32) or not np.all(np.isfinite(data)):
                raise ValueError("invalid exact E6B boundary")
            self.boundary[start:stop] = data
        self.written[start:stop] = True

    def finish(self) -> None:
        if not np.all(self.written):
            raise RuntimeError("candidate artifact does not cover every frame")
        self.handle.flush()
        self.handle.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        try:
            self.handle.close()
        finally:
            if self.temporary.exists():
                self.temporary.unlink()


def _chunk_stop(cursor: int, stop: int, *, chunk_frames: int, split: str) -> int:
    candidate = min(cursor + chunk_frames, stop)
    if split == "validation":
        block_stop = (
            VALIDATION_INTERVAL[0]
            + ((cursor - VALIDATION_INTERVAL[0]) // VALIDATION_BLOCK_FRAMES + 1)
            * VALIDATION_BLOCK_FRAMES
        )
        candidate = min(candidate, block_stop)
    return candidate


def evaluate_split(
    *,
    split: str,
    frames: tuple[int, ...],
    family: str,
    seed: int,
    codec: str,
    checkpoint_sha256: str,
    model: torch.nn.Module,
    catalog,
    device: torch.device,
    chunk_frames: int,
    candidate_path: Path,
) -> dict[str, Any]:
    spec = native_view_spec(family)
    overall = MatchedCodecAccumulator(spec=spec)
    blocks = (
        [MatchedCodecAccumulator(spec=spec) for _ in range(8)]
        if split == "validation"
        else []
    )
    dataset = CodecFrameDataset(
        catalog,
        family=family,
        split=split,
        frames=frames,
        augment=False,
        seed=seed,
        return_physical=True,
    )
    writer = CandidateWriter(
        candidate_path,
        family=family,
        frames=frames,
        codec=codec,
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
    )
    fields = FAMILY_FIELDS[family]
    started = time.monotonic()
    try:
        cursor = frames[0]
        while cursor < frames[-1] + 1:
            stop = _chunk_stop(
                cursor,
                frames[-1] + 1,
                chunk_frames=chunk_frames,
                split=split,
            )
            items = [dataset[index - frames[0]] for index in range(cursor, stop)]
            frame_indices = np.asarray(
                [item["frame_index"] for item in items], dtype=np.int64
            )
            if not np.array_equal(frame_indices, np.arange(cursor, stop)):
                raise ValueError("dataset returned non-chronological frames")
            standardized_truth = np.stack(
                [item["volume"] for item in items], axis=0
            )
            physical_truth = np.stack(
                [item["physical_volume"] for item in items], axis=0
            )
            with torch.inference_mode():
                tensor = torch.from_numpy(standardized_truth).to(
                    device=device,
                    dtype=torch.float32,
                )
                reconstruction, _ = model(tensor)
                standardized_reconstruction = (
                    reconstruction.detach().to("cpu", torch.float32).numpy()
                )
            physical_reconstruction = decode_physical_batch(
                catalog.normalization,
                fields,
                standardized_reconstruction,
            )
            overall.update(
                standardized_truth,
                standardized_reconstruction,
                physical_truth,
                physical_reconstruction,
            )
            if split == "validation":
                block = (cursor - VALIDATION_INTERVAL[0]) // VALIDATION_BLOCK_FRAMES
                blocks[block].update(
                    standardized_truth,
                    standardized_reconstruction,
                    physical_truth,
                    physical_reconstruction,
                )
            boundary = None
            if family == "e6b":
                boundary = np.stack(
                    [item["physical_boundary"] for item in items], axis=0
                )
            writer.write(
                cursor - frames[0],
                native81_candidate_fields(family, physical_reconstruction),
                boundary,
            )
            cursor = stop
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    finally:
        dataset.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "frames": [frames[0], frames[-1] + 1],
        "frame_count": len(frames),
        "overall": overall.finalize(),
        "blocks": [block.finalize() for block in blocks],
        "candidate_native81": {
            "path": str(candidate_path),
            "sha256": sha256_path(candidate_path),
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def verify_reload_probe(
    *,
    model: torch.nn.Module,
    payload: dict[str, Any],
    catalog,
    family: str,
    seed: int,
    device: torch.device,
) -> bool:
    probe = payload["reload_probe"]
    frame = int(probe["frame_index"])
    dataset = CodecFrameDataset(
        catalog,
        family=family,
        split="validation",
        frames=[frame],
        augment=False,
        seed=seed,
    )
    try:
        target = torch.from_numpy(dataset[0]["volume"])[None].to(
            device=device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            reconstruction, latent = model(target)
        exact = torch.equal(
            reconstruction.cpu(), probe["reconstruction"]
        ) and torch.equal(latent.cpu(), probe["latent"])
        return bool(exact)
    finally:
        dataset.close()


def main() -> int:
    args = parse_args()
    if not 1 <= args.chunk_frames <= VALIDATION_BLOCK_FRAMES:
        raise ValueError("chunk-frames must lie in 1..16")
    for path in (args.checkpoint, args.training_result, args.artifact_root, args.output):
        assert_development_path(path)
    verify_checkout(args.evaluation_commit)
    output = args.output.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation {output}")
    output.mkdir(parents=True)

    checkpoint = args.checkpoint.resolve(strict=True)
    training_result_path = args.training_result.resolve(strict=True)
    training_result = load_strict_json(training_result_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = validate_selected_checkpoint(
        checkpoint_path=checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        payload=payload,
        training_result_path=training_result_path,
        training_result_sha256=args.training_result_sha256,
        training_result=training_result,
        codec=args.codec,
        family=args.family,
        seed=args.seed,
    )
    catalog = load_official_catalog(args.artifact_root)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("matched codec reconstruction requires a CUDA worker")
    torch.cuda.set_device(device)
    determinism = configure_determinism(device)
    model = restore_selected_codec(
        payload=payload,
        codec=args.codec,
        family=args.family,
        device=device,
    )
    reload_exact = verify_reload_probe(
        model=model,
        payload=payload,
        catalog=catalog,
        family=args.family,
        seed=args.seed,
        device=device,
    )
    if not reload_exact:
        raise RuntimeError("evaluation reload differs from the saved checkpoint probe")

    torch.cuda.reset_peak_memory_stats(device)
    started = datetime.now(timezone.utc)
    train = evaluate_split(
        split="train",
        frames=tuple(range(*TRAIN_INTERVAL)),
        family=args.family,
        seed=args.seed,
        codec=args.codec,
        checkpoint_sha256=args.checkpoint_sha256,
        model=model,
        catalog=catalog,
        device=device,
        chunk_frames=args.chunk_frames,
        candidate_path=output / "train_candidate_native81.h5",
    )
    validation = evaluate_split(
        split="validation",
        frames=tuple(range(*VALIDATION_INTERVAL)),
        family=args.family,
        seed=args.seed,
        codec=args.codec,
        checkpoint_sha256=args.checkpoint_sha256,
        model=model,
        catalog=catalog,
        device=device,
        chunk_frames=args.chunk_frames,
        candidate_path=output / "validation_candidate_native81.h5",
    )
    materiality = training_materiality(train["overall"])
    native_gate = build_matched_o1_view_gate(
        validation_overall=validation["overall"],
        validation_blocks=validation["blocks"],
        materiality=materiality,
    )
    result = {
        "schema_version": 1,
        "scope": "phase2_matched_o1_codec_reconstruction",
        "status": "completed_pending_exact_elliptic_and_transport",
        "development_run": "85604",
        "held_out_85606_read": False,
        "evaluation_commit": args.evaluation_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "identity": identity.to_record(),
        "checkpoint_reload_bitwise_exact_on_evaluation_worker": reload_exact,
        "native_view": native_view_spec(args.family).to_record(),
        "training": train,
        "training_materiality": materiality,
        "validation": validation,
        "native_view_pretransport_gate": native_gate,
        "downstream": {
            "common_view": (
                "same_as_native_pending_transport"
                if args.family == "c5p"
                else "pending_exact_BOUT_elliptic_phi"
            ),
            "authoritative_native81_transport": "pending",
            "complete_O1_decision_allowed": False,
        },
        "execution": {
            "started_at_utc": started.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "chunk_frames": args.chunk_frames,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_process_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "determinism": determinism,
        },
        "provenance": {
            "artifact_root": str(args.artifact_root),
            "artifact_manifest_sha256": sha256_path(
                args.artifact_root / "model_dataset_manifest.json"
            ),
            "normalization_sha256": sha256_path(
                args.artifact_root / "normalization.json"
            ),
            "reconstruction_tool_sha256": sha256_path(Path(__file__).resolve()),
        },
    }
    write_strict_json_atomic(output / "result.json", result)
    print(f"wrote {output / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
