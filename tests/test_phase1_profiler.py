from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_profiler():
    path = ROOT / "paper0/tools/profile_85604_protocol.py"
    spec = importlib.util.spec_from_file_location("profile_85604_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiler = load_profiler()


class RawMetadataTests(unittest.TestCase):
    def test_raw_units_timeline_and_state_candidates_are_read_from_attrs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BOUT.dmp.0.nc"
            times = np.arange(624, dtype=np.float64) * 300.0 + 285000.0
            with h5py.File(path, "w") as handle:
                handle.create_dataset("t_array", data=times)
                for name, value in {
                    "zperiod": 5,
                    "ZMIN": 0.0,
                    "ZMAX": 0.2,
                    "Omega_ci": 95788333.03066081,
                }.items():
                    handle.create_dataset(name, data=value)
                expected_meta = {
                    "Ne": ("m^-3", 1e19),
                    "Te": ("eV", 50.0),
                    "Ti": ("eV", 50.0),
                    "phi": ("V", 50.0),
                    "Vi": ("m / s", 69205.61141651045),
                }
                for field, (units, conversion) in expected_meta.items():
                    dataset = handle.create_dataset(
                        field, shape=(624, 1, 1, 5), dtype=np.float32
                    )
                    dataset.attrs["units"] = units
                    # The actual BOUT output stores this scalar in a length-one
                    # attribute array, which must be unwrapped explicitly.
                    dataset.attrs["conversion"] = np.asarray([conversion])
                    dataset.attrs["object_reference"] = dataset.ref
                handle.create_dataset("Vort", shape=(624, 1, 1, 5))

            expected = {
                "c5_fields": list(expected_meta),
                "additional_state_candidates": ["Vort", "Ve"],
                "raw_field_metadata": {
                    field: {"units": units, "conversion": conversion}
                    for field, (units, conversion) in expected_meta.items()
                },
                "total_frames": 624,
                "raw_time_first": 285000.0,
                "raw_time_last": 471900.0,
                "normalized_frame_step": 300.0,
                "omega_ci_per_second": 95788333.03066081,
                "cadence_microseconds": 3.131905426352636,
                "zperiod": 5,
                "zmin": 0.0,
                "zmax": 0.2,
            }
            result = profiler.inspect_raw(path, expected, times)
            self.assertEqual(result["toroidal_domain"]["mode_mapping"], "n = 5k")
            self.assertIn("Vort", result["field_metadata"])
            self.assertEqual(result["missing_additional_candidates"], ["Ve"])
            self.assertTrue(result["required_field_checks"]["phi"]["units_match"])
            reference = result["field_metadata"]["Ne"]["attrs"]["object_reference"]
            self.assertEqual(reference["hdf5_reference_type"], "object")
            # The complete metadata record must remain strict JSON.
            json.dumps(result, allow_nan=False)

    def test_bout_settings_flags_are_scoped_to_the_correct_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BOUT.settings"
            path.write_text(
                """[e]
type = evolve_density, evolve_pressure, evolve_momentum

[i]
type = quasineutral, evolve_pressure, evolve_momentum

[run]
revision = abc123
version = 5.2.1

[vorticity]
diamagnetic_polarisation = true
exb_advection_simplified = false
""",
                encoding="utf-8",
            )
            result = profiler.inspect_bout_settings(
                path, {"bout_version": "5.2.1", "bout_revision": "abc123"}
            )
            self.assertTrue(result["electron_momentum_evolved"])
            self.assertTrue(result["vorticity_component_present"])
            self.assertTrue(result["diamagnetic_polarisation_enabled"])
            self.assertTrue(result["exb_advection_simplified_false"])


if __name__ == "__main__":
    unittest.main()
