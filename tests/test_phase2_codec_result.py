from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "paper0/results/phase2_o1_codec_6890650.json"
EXPECTED_RAW_SHA256 = (
    "d9440ecf7182c434976b67a33118d8c3dcb81b0fcec9a16f89745a5398aa850e"
)
EXPECTED_COMMIT = "2bf810ff226641ac1955367a18bd492ab08c442c"


class CodecOracleResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_provenance_and_blind_test_exclusion(self) -> None:
        result = self.result
        self.assertEqual(
            result["result_type"],
            "phase2_o1_codec_reconstruction_oracle_compact",
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["scope"]["run_id"], "85604")
        self.assertFalse(result["scope"]["shot_85606_accessed"])
        self.assertEqual(result["execution"]["paper0_commit"], EXPECTED_COMMIT)
        self.assertEqual(str(result["execution"]["slurm_job_id"]), "6890650")
        self.assertEqual(result["raw_artifact"]["sha256"], EXPECTED_RAW_SHA256)
        self.assertRegex(
            result["raw_artifact"]["sha256"], re.compile(r"^[0-9a-f]{64}$")
        )

    def test_mode_mapping_and_curve_extent_are_explicit(self) -> None:
        result = self.result
        self.assertEqual(result["scope"]["zperiod"], 5)
        self.assertEqual(result["scope"]["mode_mapping"], "n = 5k")
        for codec_name in ("f8", "z44"):
            curves = result["codec_results"][codec_name]["mode_curves_k0_to_k16"]
            self.assertEqual(curves["stored_k"], list(range(17)))
            self.assertEqual(curves["full_torus_n"], [5 * k for k in range(17)])

    def test_locked_outcomes_are_preserved(self) -> None:
        result = self.result
        f8 = result["codec_results"]["f8"]
        z44 = result["codec_results"]["z44"]
        self.assertEqual(f8["preliminary_gate"]["preliminary_status"], "fail")
        self.assertEqual(z44["preliminary_gate"]["preliminary_status"], "fail")
        self.assertTrue(f8["preliminary_gate"]["field_reconstruction"]["pass"])
        self.assertTrue(f8["preliminary_gate"]["cross_field"]["pass"])
        self.assertFalse(f8["preliminary_gate"]["spectral_transfer"]["pass"])
        self.assertFalse(z44["preliminary_gate"]["spectral_transfer"]["pass"])
        self.assertFalse(z44["preliminary_gate"]["cross_field"]["pass"])
        self.assertAlmostEqual(
            f8["aggregate_five_field_rmse_legacy_standardized"],
            0.024922855480884624,
        )
        self.assertAlmostEqual(
            z44["aggregate_five_field_rmse_legacy_standardized"],
            0.032789490059632824,
        )

    def test_compact_result_itself_is_stable_json(self) -> None:
        digest = hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest()
        self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))
        self.assertLess(RESULT_PATH.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
