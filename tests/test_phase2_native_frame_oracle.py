from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper0" / "tools"
EXTRACTOR = TOOLS / "extract_native_85604_frames.py"
COMPARATOR = TOOLS / "compare_hermes_native_frame_oracle.py"
LAUNCHER = ROOT / "cluster" / "phase2_hermes_native_frame_oracle.sbatch"
ORACLE_DIR = ROOT / "paper0" / "oracles" / "hermes_native_frames"
MANIFEST = ROOT / "paper0" / "manifests" / "phase2_native_frame_oracle.json"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


EXTRACT = load_module("paper0_native_extract", EXTRACTOR)
COMPARE = load_module("paper0_native_compare", COMPARATOR)


class NativeFrameOracleImplementationTests(unittest.TestCase):
    def test_launcher_is_cpu_only_clean_locked_and_syntax_valid(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--ntasks=4",
            "--no-requeue",
            "Refusing to overwrite",
            "920ba829cc78cdab0dbf6101c69fecc4689bd8dd",
            "458eeecbd6da1afb882d0de2b652271fc2c2ca142c39a636a52f3adc5c16ef3f",
            "3c766083078ec17d737a7ac595868adf1706e0596a9e614bb3ac73f071c1834d",
            "d3abc5e32cdad3ea9c42faf432dcaed465070ee5792cb62e550b9baaad9953e6",
            manifest_digest,
            "--atol 5e-10",
            "--rtol 5e-10",
            "extract_native_85604_frames.py",
            "paper0:input_file=",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_gpl_driver_loads_frozen_frames_and_source_terms(self) -> None:
        source = (
            ORACLE_DIR / "hermes_native_frames_oracle.cxx"
        ).read_text(encoding="utf-8")
        options = (ORACLE_DIR / "BOUT.inp").read_text(encoding="utf-8")
        cmake = (ORACLE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: GPL-3.0-or-later", source)
        self.assertIn("EXPECTED_FRAMES{0, 156, 312, 467, 623}", source)
        self.assertIn('ADVECTED_FIELDS{"Ne", "Pe", "Pi"}', source)
        self.assertIn("nc_get_vara_double", source)
        self.assertIn("mesh->getYProcIndex()", source)
        self.assertIn("mesh->communicate(field)", source)
        self.assertIn("corner_plus - corner_minus", source)
        self.assertIn("flow_xz + flow_xy", source)
        self.assertIn("coord->J(i, j) * coord->dx(i, j)", source)
        self.assertIn("BoutComm::size() != 4", source)
        self.assertIn("zperiod = 5", options)
        self.assertIn("NXPE = 1", options)
        self.assertIn("netCDF::netcdf", cmake)

    def test_frozen_selection_and_rank_filename_parsing(self) -> None:
        self.assertEqual(
            EXTRACT.nearest_half_up_indices(624, [0.0, 0.25, 0.5, 0.75, 1.0]),
            [0, 156, 312, 467, 623],
        )
        self.assertEqual(EXTRACT.rank_number(Path("BOUT.dmp.255.nc")), 255)
        with self.assertRaises(ValueError):
            EXTRACT.rank_number(Path("BOUT.dmp.bad.nc"))
        with self.assertRaises(ValueError):
            EXTRACT.nearest_half_up_indices(624, [-0.1])

    def test_rank_blocks_are_placed_by_processor_coordinates(self) -> None:
        destination = np.full((2, 4, 6, 3), np.nan)
        block = np.arange(2 * 2 * 2 * 3, dtype=np.float64).reshape(2, 2, 2, 3)
        EXTRACT.place_rank_block(
            destination,
            block,
            pe_x=1,
            pe_y=2,
            mxsub=2,
            mysub=2,
        )
        np.testing.assert_array_equal(destination[:, 2:4, 4:6], block)
        self.assertTrue(np.all(np.isnan(destination[:, :2])))
        with self.assertRaises(ValueError):
            EXTRACT.place_rank_block(
                destination,
                block,
                pe_x=2,
                pe_y=0,
                mxsub=2,
                mysub=2,
            )

    def test_array_digest_locks_shape_dtype_and_values(self) -> None:
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        self.assertEqual(
            EXTRACT.sha256_array(values),
            EXTRACT.sha256_array(values.astype(np.float64)),
        )
        self.assertNotEqual(
            EXTRACT.sha256_array(values),
            EXTRACT.sha256_array(values.reshape(4, 3, 2)),
        )
        changed = values.copy()
        changed[0, 0, 0] = 1.0
        self.assertNotEqual(
            EXTRACT.sha256_array(values), EXTRACT.sha256_array(changed)
        )

    def test_noncollapse_gates_are_strict_and_finite(self) -> None:
        varying = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        self.assertTrue(COMPARE.dynamic_input_metrics(varying)["passed"])
        self.assertFalse(
            COMPARE.dynamic_input_metrics(np.ones_like(varying))["passed"]
        )
        contaminated = varying.copy()
        contaminated[0, 0, 0] = np.nan
        self.assertFalse(COMPARE.dynamic_input_metrics(contaminated)["passed"])
        mask = np.ones((2, 3), dtype=bool)
        self.assertTrue(COMPARE.maximum_absolute_metrics(varying, mask)["passed"])
        self.assertFalse(
            COMPARE.maximum_absolute_metrics(np.zeros_like(varying), mask)[
                "passed"
            ]
        )

    def test_implementation_constants_match_frozen_manifest(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(list(COMPARE.FRAME_INDICES), manifest["frame_selection"]["indices"])
        self.assertEqual(
            list(COMPARE.ADVECTED_FIELDS),
            ["Ne", "Pe", "Pi"],
        )
        closure = manifest["closure_acceptance"]
        operator = manifest["operator_acceptance"]
        self.assertEqual(COMPARE.CLOSURE_ATOL, closure["atol"])
        self.assertEqual(COMPARE.CLOSURE_RTOL, closure["rtol"])
        self.assertEqual(COMPARE.DEFAULT_ATOL, operator["continuous_atol"])
        self.assertEqual(COMPARE.DEFAULT_RTOL, operator["continuous_rtol"])
        self.assertEqual(
            COMPARE.CONSERVATION_ATOL, operator["conservation_atol"]
        )
        self.assertEqual(
            COMPARE.CONSERVATION_RTOL, operator["conservation_rtol"]
        )


if __name__ == "__main__":
    unittest.main()
