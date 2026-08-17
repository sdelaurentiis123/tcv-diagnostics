from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o1_codec_transport.sbatch"


class CodecTransportLauncherTests(unittest.TestCase):
    def test_launcher_is_rocky9_clean_commit_and_85604_only(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        required_digests = {
            "843f9ae99d08fbcdabce977b53e4f6b49be05641a82a387d100b237224b77777",
            "a17b536856c6b8108c0553c300200e074e41407129e47ef402a4de51882ea1ba",
            "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817",
            "eed18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43",
            "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
            "957464ad2510ac47bdf882d419db3061fa60d4455d6bc5242bc05baa9a410bf6",
            "1191f439f8b58fdb38d570a5ac6b54abed12c06b8455b634c5063d787fe2b357",
            "0f67375309e8c9d12118ab0a8acc5430f26ae1ee2e633f06f96e0140af3f076c",
            "36e2e80668ccbc121c31e25595a43bc18b02d903d5122f74f11e64e8a8a52678",
            "3fb6e6be7649e86fc0626f5d847adf13649e213c82b543c714ae258332bfdf7d",
        }
        for digest in required_digests:
            self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))
            self.assertIn(digest, text)
        self.assertIn("VERSION_ID%%.*", text)
        self.assertIn("status --porcelain --untracked-files=all", text)
        self.assertIn("evaluate_codec_transport_oracle.py", text)
        self.assertIn("--chunk-frames 4", text)
        self.assertIn("--device cuda", text)
        self.assertIn("TCV_85604_train.hdf5", text)
        self.assertIn("TCV_c5_train.hdf5", text)
        self.assertNotIn("85606", text)
        self.assertNotRegex(text, re.compile(r"/test(?:/|\s|\")"))


if __name__ == "__main__":
    unittest.main()
