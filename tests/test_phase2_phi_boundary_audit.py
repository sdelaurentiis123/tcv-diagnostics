from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper0" / "tools" / "audit_85604_phi_boundary_state.py"
SPEC = importlib.util.spec_from_file_location("phi_boundary_audit", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load phi boundary audit")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
MANIFEST = json.loads(
    (ROOT / "paper0" / "manifests" / "phase2_85604_phi_boundary_state.json").read_text(
        encoding="utf-8"
    )
)


class FakePhi:
    dimensions = ("t", "x", "y", "z")
    shape = (624, 8, 6, 81)
    dtype = np.dtype("float64")
    cell_location = "CELL_CENTRE"
    time_dimension = "t"
    source = "vorticity"
    units = "V"
    conversion = 50.0


class PhiBoundaryAuditStructureTests(unittest.TestCase):
    def test_phi_metadata_validation_records_exact_source_contract(self) -> None:
        result = AUDIT.validate_phi_metadata(FakePhi(), MANIFEST["phi_metadata"])
        self.assertEqual(result["shape"], [624, 8, 6, 81])
        self.assertEqual(result["source"], "vorticity")
        self.assertEqual(result["conversion_volts"], 50.0)

    def test_phi_metadata_validation_rejects_axis_or_conversion_drift(self) -> None:
        variable = FakePhi()
        variable.dimensions = ("t", "y", "x", "z")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            AUDIT.validate_phi_metadata(variable, MANIFEST["phi_metadata"])
        variable.dimensions = ("t", "x", "y", "z")
        variable.conversion = 49.0
        with self.assertRaisesRegex(ValueError, "conversion"):
            AUDIT.validate_phi_metadata(variable, MANIFEST["phi_metadata"])

    def test_path_guard_refuses_blind_or_ambiguous_root(self) -> None:
        AUDIT.verify_development_path(Path("/data/85604"))
        with self.assertRaisesRegex(ValueError, "held-out"):
            AUDIT.verify_development_path(Path("/data/85606"))
        with self.assertRaisesRegex(ValueError, "must identify"):
            AUDIT.verify_development_path(Path("/data/unknown"))

    def test_stream_digest_locks_name_shape_dtype_and_values(self) -> None:
        values = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
        first = AUDIT.sha256_array("inner:adjacent", values)
        second = AUDIT.sha256_array("inner:adjacent", values.copy())
        changed_name = AUDIT.sha256_array("outer:adjacent", values)
        changed_value = AUDIT.sha256_array("inner:adjacent", values + 1.0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed_name)
        self.assertNotEqual(first, changed_value)


if __name__ == "__main__":
    unittest.main()
