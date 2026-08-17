from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o1_codec_oracle.sbatch"


class CodecLauncherTests(unittest.TestCase):
    def test_launcher_is_locked_to_expected_85604_inputs_and_codecs(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        required_digests = {
            "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817",
            "eed18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43",
            "66509d2b0c9a1aaa03959e0e33691d443f39fa24bbad93a0dbb41e291176e776",
            "9f65dc523b8ee32ea5dd87842b99075de15f9aae86d2e71a5da55bc37091a44e",
            "5d868c1cfc5a17ce26c2f6ce86ced50d7b55525c6967c5b599b1074058b67284",
            "095d25f9b6e867103d4cfb946cc9ea8a172a5a6db5b28e5726428c4c57e4979d",
            "5c20d880799301a636ff8de67d34d39221d9c5a7e9e0bc2123ae84ee43fc5c83",
            "def0e35e3a97a31627415186130b5b8ac6bf69611dc50caacafb73398a706bc5",
            "a56aef6d1d04c86af91238eeb51d93d345dce5a74481dde5ae3dc244f842691f",
            "281f8541aa09822147f8769e9a11fb63497aa54783dd9a806b173e76c5fbaede",
            "ad6aab36d52ea7aba2a0c45006a33413304c0d9ceb9abffd52a497a24adf616f",
            "3fb6e6be7649e86fc0626f5d847adf13649e213c82b543c714ae258332bfdf7d",
        }
        for digest in required_digests:
            self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))
            self.assertIn(digest, text)

        self.assertIn("TCV_c5_train.hdf5", text)
        self.assertIn("TCV_c5_valid.hdf5", text)
        self.assertIn("w24x2ybf_tcv_c5_dcae_3d_tcv_f8c64", text)
        self.assertIn("z44c6604191_tcv_c5_dcae_3d_tcv_f8z2c64", text)
        self.assertIn("evaluate_codec_oracle.py", text)
        self.assertIn("--chunk-frames 4", text)
        self.assertIn("VERSION_ID%%.*", text)
        self.assertIn("status --porcelain --untracked-files=all", text)
        self.assertNotIn("85606", text)
        self.assertNotRegex(text, re.compile(r"/test(?:/|\s|\")"))


if __name__ == "__main__":
    unittest.main()
