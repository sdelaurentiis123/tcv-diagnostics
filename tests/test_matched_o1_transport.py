"""Known-answer tests for immutable matched O1 native truth access."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import h5py
import numpy as np

from tcv_diagnostics.matched_o1_transport import (
    NATIVE_SHAPE,
    NATIVE_TRUTH_FIELDS,
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
