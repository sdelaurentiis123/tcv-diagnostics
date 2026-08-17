"""Known-answer tests for immutable matched O1 native truth access."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import h5py
import numpy as np

from tcv_diagnostics.matched_o1_transport import (
    CANDIDATE_NATIVE_FIELDS,
    E6B_COMMON_COMPONENTS,
    MODEL_SHAPE,
    NATIVE_SHAPE,
    NATIVE_TRUTH_FIELDS,
    MatchedCandidateArtifact,
    MatchedPhiArtifact,
    NativeTruthCatalog,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_truth(root: Path) -> dict:
    records = []
    for index in range(8):
        start = index * 78
        path = root / f"canonical_shard_{index}.nc"
        with h5py.File(path, "x") as handle:
            handle.create_dataset(
                "frame_index", data=np.arange(start, start + 78, dtype=np.int64)
            )
            for field_index, field in enumerate(NATIVE_TRUTH_FIELDS):
                handle.create_dataset(
                    field,
                    shape=(78, *NATIVE_SHAPE),
                    dtype="f8",
                    chunks=(1, *NATIVE_SHAPE),
                    fillvalue=float(field_index + 1),
                )
            handle.create_dataset(
                "saved_midpoint",
                shape=(78, 2, 32),
                dtype="f8",
                chunks=(1, 2, 32),
                fillvalue=0.25,
            )
        records.append(
            {
                "shard_index": index,
                "start": start,
                "stop": start + 78,
                "canonical_file": str(path),
                "canonical_file_sha256": _sha256(path),
            }
        )
    return {
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "decision": {"all_frame_bidirectional_closure_validated": True},
        "ordered_gates": {f"G{index}": True for index in range(5)},
        "extraction": {"canonical_shards": records},
    }


def _candidate(root: Path, family: str, frames: tuple[int, ...]) -> Path:
    path = root / f"{family}_candidate.h5"
    with h5py.File(path, "x") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["development_run"] = "85604"
        handle.attrs["held_out_85606_read"] = False
        handle.attrs["family"] = family
        handle.attrs["codec"] = "dcae_l20"
        handle.attrs["seed"] = 1701
        handle.attrs["checkpoint_sha256"] = "checkpoint"
        handle.attrs["zperiod"] = 5
        handle.attrs["native_shape"] = NATIVE_SHAPE
        coordinates = handle.create_group("coordinates")
        coordinates.create_dataset("frame_index", data=frames)
        candidate = handle.create_group("candidate")
        for index, field in enumerate(CANDIDATE_NATIVE_FIELDS[family]):
            candidate.create_dataset(
                field,
                shape=(len(frames), *NATIVE_SHAPE),
                dtype="f4",
                chunks=(1, *NATIVE_SHAPE),
                fillvalue=float(index + 1),
            )
        if family == "e6b":
            boundary = handle.create_group("boundary").create_dataset(
                "Bphi",
                shape=(len(frames), 2, 32),
                dtype="f4",
                chunks=(1, 2, 32),
                fillvalue=0.25,
            )
            boundary.attrs["policy"] = "exact_bypass_from_model_dataset"
            model = handle.create_group("model88")
            for index, field in enumerate(E6B_COMMON_COMPONENTS):
                model.create_dataset(
                    field,
                    shape=(len(frames), *MODEL_SHAPE),
                    dtype="f4",
                    chunks=(1, *MODEL_SHAPE),
                    fillvalue=float(index + 1),
                )
    return path


def test_native_truth_catalog_crosses_shards_without_overlap() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        catalog = NativeTruthCatalog(_compact_truth(Path(temporary)))
        values = catalog.read(77, 79, fields=("Ne", "phi"))
        assert values["Ne"].shape == (2, *NATIVE_SHAPE)
        assert values["phi"].shape == (2, *NATIVE_SHAPE)
        np.testing.assert_array_equal(values["Ne"], 1.0)
        np.testing.assert_array_equal(values["phi"], 5.0)


def test_native_truth_catalog_refuses_failed_source_gate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        result = _compact_truth(Path(temporary))
        result["ordered_gates"]["G3"] = False
        try:
            NativeTruthCatalog(result)
        except ValueError as error:
            assert "gates did not pass" in str(error)
        else:
            raise AssertionError("failed native truth gate was accepted")


def test_native_truth_catalog_refuses_reordered_shard_record() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        result = _compact_truth(Path(temporary))
        result["extraction"]["canonical_shards"][2]["shard_index"] = 3
        try:
            NativeTruthCatalog(result)
        except ValueError as error:
            assert "order or interval" in str(error)
        else:
            raise AssertionError("reordered native truth shard was accepted")


def test_matched_candidate_artifact_verifies_and_reads_e6b_views() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = _candidate(Path(temporary), "e6b", (496, 497))
        artifact = MatchedCandidateArtifact(
            path,
            sha256=_sha256(path),
            family="e6b",
            codec="dcae_l20",
            seed=1701,
            checkpoint_sha256="checkpoint",
            frames=(496, 497),
        )
        native = artifact.read_native(497, 498)
        model = artifact.read_model88(496, 497)
        assert native["Vort"].shape == (1, *NATIVE_SHAPE)
        assert model["NVi"].shape == (1, *MODEL_SHAPE)
        np.testing.assert_array_equal(artifact.read_boundary(496, 498), 0.25)


def test_matched_phi_artifact_is_tied_to_candidate_hash() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        candidate = _candidate(root, "e6b", (496, 497))
        candidate_hash = _sha256(candidate)
        phi = root / "phi.h5"
        with h5py.File(phi, "x") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["development_run"] = "85604"
            handle.attrs["held_out_85606_read"] = False
            handle.attrs["zperiod"] = 5
            handle.attrs["truth_layout"] = False
            handle.attrs["source_input_sha256"] = candidate_hash
            handle.create_dataset("frame_index", data=(496, 497))
            handle.create_dataset(
                "phi",
                shape=(2, *NATIVE_SHAPE),
                dtype="f8",
                chunks=(1, *NATIVE_SHAPE),
                fillvalue=3.0,
            )
        artifact = MatchedPhiArtifact(
            phi,
            sha256=_sha256(phi),
            source_candidate_sha256=candidate_hash,
            frames=(496, 497),
        )
        np.testing.assert_array_equal(artifact.read(496, 498), 3.0)
        try:
            MatchedPhiArtifact(
                phi,
                sha256=_sha256(phi),
                source_candidate_sha256="wrong",
                frames=(496, 497),
            )
        except ValueError as error:
            assert "attributes differ" in str(error)
        else:
            raise AssertionError("phi artifact accepted the wrong candidate hash")

        with h5py.File(phi, "r+") as handle:
            del handle.attrs["held_out_85606_read"]
        try:
            MatchedPhiArtifact(
                phi,
                sha256=_sha256(phi),
                source_candidate_sha256=candidate_hash,
                frames=(496, 497),
            )
        except ValueError as error:
            assert "attributes are missing" in str(error)
        else:
            raise AssertionError("phi artifact accepted a missing blind-lock attribute")
